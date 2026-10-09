"""Read-only investigation before a reply is drafted.

The five tools cannot write a case. The customer lookup is bound to the envelope
sender, so an address written in the email body cannot switch the profile.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.agents.middleware.types import InputAgentState
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from pydantic import ValidationError

from app.brain import make_chat_model, wrap_untrusted
from app.completion import Completion, Usage
from app.config import Settings
from app.db import Store
from app.failures import log_failure
from app.guards import strip_delimiters
from app.prompts import INVESTIGATE_SYSTEM
from app.schemas import Fact, Findings, InboundEmail, Triage

TOOL_CAP = 5
TOOL_NAMES = (
    "get_customer_profile",
    "get_service_history",
    "find_prior_cases",
    "search_similar_cases",
    "check_recalls",
)


@dataclass
class ToolTrace:
    tool: str
    args: dict[str, Any]
    result: str
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def as_event_extra(self) -> dict[str, int]:
        extra: dict[str, int] = {}
        if self.latency_ms is not None:
            extra["latency_ms"] = self.latency_ms
        if self.input_tokens is not None:
            extra["input_tokens"] = self.input_tokens
        if self.output_tokens is not None:
            extra["output_tokens"] = self.output_tokens
        return extra


class InvestigationFailed(Exception):
    """The investigator stopped. The pipeline keeps the fallback findings."""

    def __init__(self, kind: str, findings: Findings, traces: list[ToolTrace]) -> None:
        super().__init__(kind)
        self.kind = kind
        self.findings = findings
        self.traces = traces


@dataclass(frozen=True)
class Vehicle:
    model: str
    model_year: int
    vin: str
    purchase_date: str
    mileage: int


@dataclass(frozen=True)
class Customer:
    email: str
    name: str
    vehicle: Vehicle | None
    preferred_branch: str


@dataclass(frozen=True)
class Visit:
    date: str
    branch: str
    work: str
    technician_notes: str


@dataclass(frozen=True)
class Recall:
    id: str
    model: str
    year_from: int
    year_to: int
    component: str
    remedy: str
    status: str


class Enterprise:
    def __init__(self, customers: dict[str, Customer], history: dict[str, list[Visit]], recalls: list[Recall]) -> None:
        self.customers = customers
        self.history = history
        self.recalls = recalls

    @classmethod
    def load(cls, directory: Path) -> Enterprise:
        raw_customers = json.loads((directory / "customers.json").read_text(encoding="utf-8"))
        raw_history = json.loads((directory / "service_history.json").read_text(encoding="utf-8"))
        raw_recalls = json.loads((directory / "recalls.json").read_text(encoding="utf-8"))
        customers: dict[str, Customer] = {}
        for email, row in raw_customers.items():
            vehicle = None
            if row.get("vehicle"):
                v = row["vehicle"]
                vehicle = Vehicle(v["model"], int(v["model_year"]), v["vin"], v["purchase_date"], int(v["mileage"]))
            customers[email.lower()] = Customer(email, row["name"], vehicle, row.get("preferred_branch", ""))
        history = {
            vin: [
                Visit(item["date"], item["branch"], item["work"], item.get("technician_notes", "")) for item in visits
            ]
            for vin, visits in raw_history.items()
        }
        recalls = [
            Recall(
                item["id"],
                item["model"],
                int(item["year_from"]),
                int(item["year_to"]),
                item["component"],
                item["remedy"],
                item["status"],
            )
            for item in raw_recalls
        ]
        return cls(customers, history, recalls)

    def customer(self, sender: str) -> Customer | None:
        return self.customers.get(sender.lower())


def profile_text(enterprise: Enterprise, sender: str) -> str:
    customer = enterprise.customer(sender)
    if customer is None:
        return "No customer profile for this sender."
    if customer.vehicle is None:
        return f"{customer.name}. No vehicle on file. Preferred branch {customer.preferred_branch or 'unknown'}."
    vehicle = customer.vehicle
    return (
        f"{customer.name}. Vehicle {vehicle.model} {vehicle.model_year}, "
        f"purchased {vehicle.purchase_date}, mileage {vehicle.mileage}. "
        f"Preferred branch {customer.preferred_branch}. VIN {vehicle.vin}."
    )


def history_text(enterprise: Enterprise, sender: str, vin: str) -> str:
    customer = enterprise.customer(sender)
    own = customer.vehicle.vin if customer and customer.vehicle else None
    if not own or vin.strip().upper() != own.upper():
        return "VIN is not on this sender's profile. Refusing to look it up."
    visits = enterprise.history.get(own, [])
    if not visits:
        return "No service visits for this vehicle."
    lines = []
    for visit in visits:
        notes = f" Internal notes: {visit.technician_notes}" if visit.technician_notes else ""
        lines.append(f"{visit.date} at {visit.branch}: {visit.work}.{notes}")
    return " ".join(lines)


def _model_matches(stored: str, query: str) -> bool:
    """Match a profile model name or its model word (Atlas matches Meridian Atlas)."""
    left = stored.strip().lower()
    right = query.strip().lower()
    if not left or not right:
        return False
    if left == right:
        return True
    left_name = left.split()[-1]
    right_name = right.split()[-1]
    return len(left_name) >= 3 and left_name == right_name


def recalls_text(enterprise: Enterprise, model: str, model_year: int) -> str:
    matches = [
        recall
        for recall in enterprise.recalls
        if recall.status == "open"
        and _model_matches(recall.model, model)
        and recall.year_from <= model_year <= recall.year_to
    ]
    if not matches:
        return "No open recall for this model and year."
    return " ".join(
        f"Open recall {recall.id} ({recall.component}, {recall.year_from}-{recall.year_to}). Remedy: {recall.remedy}."
        for recall in matches
    )


def _current_created_at(store: Store, case_id: str | None) -> str | None:
    if not case_id:
        return None
    row = store.get_case(case_id)
    if row is None:
        return None
    return str(row["created_at"])


def _kept_case(row: dict[str, Any], case_id: str | None, created_at: str | None) -> bool:
    """Drop this case, and any case opened after it, so a lookup cannot see the future or itself."""
    if case_id and str(row.get("id")) == case_id:
        return False
    return not (created_at and str(row.get("created_at") or "") > created_at)


def prior_cases_text(store: Store, sender: str, case_id: str | None = None) -> str:
    created_at = _current_created_at(store, case_id)
    rows = [
        row
        for row in store.list_cases(limit=100)
        if str(row["sender"]).lower() == sender.lower() and _kept_case(row, case_id, created_at)
    ]
    if not rows:
        return "No earlier cases from this sender."
    payload = [
        {
            "id": str(row["id"]),
            "subject": strip_delimiters(str(row.get("subject") or "")),
            "category": row.get("category"),
            "severity": row.get("severity"),
            "status": row.get("status"),
        }
        for row in rows[:5]
    ]
    return json.dumps(payload, ensure_ascii=False)


def similar_cases_text(
    store: Store,
    sender: str,
    query: str,
    category: str | None,
    case_id: str | None = None,
) -> str:
    created_at = _current_created_at(store, case_id)
    rows = store.list_cases(category=category, text=query, limit=20)
    others = [
        row for row in rows if str(row["sender"]).lower() != sender.lower() and _kept_case(row, case_id, created_at)
    ][:5]
    if not others:
        return "No similar cases."
    return json.dumps(
        [
            {
                "id": row.get("id"),
                "subject": strip_delimiters(str(row.get("subject") or "")),
                "category": row.get("category"),
                "severity": row.get("severity"),
                "status": row.get("status"),
            }
            for row in others
        ],
        ensure_ascii=False,
    )


def make_tools(
    enterprise: Enterprise,
    store: Store,
    sender: str,
    traces: list[ToolTrace],
    case_id: str | None = None,
) -> list[Any]:
    """Tools closed over the envelope sender. The sixth call does not run a lookup."""

    def guarded(name: str, args: dict[str, Any], fn: Any) -> str:
        if len(traces) >= TOOL_CAP:
            return f"Tool cap of {TOOL_CAP} reached. Stop and report the findings you already have."
        started = time.perf_counter()
        try:
            result = str(fn())
        except Exception as exc:  # noqa: BLE001  # a lookup must not leak exception text to the model
            log_failure(case_id or "-", name, exc)
            result = "lookup failed"
        traces.append(ToolTrace(name, args, result, latency_ms=round((time.perf_counter() - started) * 1000)))
        return result

    @tool
    def get_customer_profile() -> str:
        """Profile for the envelope sender. Takes no address. Ignores any address in the email."""
        return guarded("get_customer_profile", {}, lambda: profile_text(enterprise, sender))

    @tool
    def get_service_history(vin: str) -> str:
        """Service visits for a VIN. Only the VIN on this sender's own profile is allowed."""
        return guarded("get_service_history", {"vin": vin}, lambda: history_text(enterprise, sender, vin))

    @tool
    def find_prior_cases() -> str:
        """Earlier ResolveDesk cases from this same sender."""
        return guarded("find_prior_cases", {}, lambda: prior_cases_text(store, sender, case_id))

    @tool
    def search_similar_cases(query: str, category: str | None = None) -> str:
        """Up to 5 other customers' cases by text and optional category."""
        return guarded(
            "search_similar_cases",
            {"query": query, "category": category},
            lambda: similar_cases_text(store, sender, query, category, case_id),
        )

    @tool
    def check_recalls() -> str:
        """Open recalls for the vehicle on this sender's profile. Takes no model or year."""

        def lookup() -> str:
            customer = enterprise.customer(sender)
            if customer is None or customer.vehicle is None:
                return "No vehicle on file"
            return recalls_text(enterprise, customer.vehicle.model, customer.vehicle.model_year)

        return guarded("check_recalls", {}, lookup)

    return [get_customer_profile, get_service_history, find_prior_cases, search_similar_cases, check_recalls]


def _short(text: str, limit: int = 160) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def trace_detail(trace: ToolTrace) -> str:
    if trace.args:
        rendered = ", ".join(f"{key}={value}" for key, value in trace.args.items())
        head = f"{trace.tool}({rendered})"
    else:
        head = f"{trace.tool}()"
    return f"{head} → {_short(trace.result)}"


def _facts_from_traces(traces: list[ToolTrace]) -> list[Fact]:
    facts: list[Fact] = []
    for trace in traces:
        if trace.result.startswith(("No ", "VIN is not", "error:", "lookup failed")):
            continue
        if trace.tool == "get_customer_profile":
            # The handler can see the profile. The VIN stays out of the fact the drafter quotes.
            text = trace.result.split(" VIN ")[0].strip()
            facts.append(Fact(text=text, source_tool=trace.tool))
        elif trace.tool == "get_service_history":
            for chunk in trace.result.split(". "):
                if chunk.strip().startswith("Internal notes"):
                    continue
                work = chunk.split(" Internal notes:")[0].strip().rstrip(".")
                if work:
                    facts.append(Fact(text=work, source_tool=trace.tool))
        elif trace.tool == "check_recalls" or trace.tool == "find_prior_cases":
            facts.append(Fact(text=trace.result, source_tool=trace.tool))
    return facts


def apply_tool_facts(findings: Findings, traces: list[ToolTrace]) -> tuple[Findings, list[str]]:
    """Keep the model summary. Facts are the tool text, not whatever the model wrote."""
    called = {trace.tool for trace in traces}
    notes: list[str] = []
    for fact in findings.facts:
        if fact.source_tool not in called:
            notes.append(f"fact removed: {fact.text} (tool not called)")
            continue
        if not any(fact.text in trace.result for trace in traces if trace.tool == fact.source_tool):
            notes.append(f"fact removed: {fact.text} (not in tool result)")
    return findings.model_copy(update={"facts": _facts_from_traces(traces)}), notes


def _nothing_relevant(facts: list[Fact]) -> bool:
    return not any(fact.source_tool in {"get_service_history", "check_recalls", "find_prior_cases"} for fact in facts)


class Investigator:
    def __init__(
        self, settings: Settings, enterprise: Enterprise, store: Store, *, offline: bool | None = None
    ) -> None:
        self._settings = settings
        self._enterprise = enterprise
        self._store = store
        self._offline = settings.llm_provider == "offline" if offline is None else offline

    def investigate(
        self, email: InboundEmail, triage: Triage, case_id: str | None = None
    ) -> tuple[Findings, list[ToolTrace]] | Completion[tuple[Findings, list[ToolTrace]]]:
        if self._offline:
            return self._offline_run(email, triage, case_id)
        started = time.perf_counter()
        findings, traces = self.run_agent(make_chat_model(self._settings), email, triage, case_id)
        input_tokens = [trace.input_tokens for trace in traces if trace.input_tokens is not None]
        output_tokens = [trace.output_tokens for trace in traces if trace.output_tokens is not None]
        usage = Usage(
            latency_ms=round((time.perf_counter() - started) * 1000),
            input_tokens=sum(input_tokens) if input_tokens else None,
            output_tokens=sum(output_tokens) if output_tokens else None,
        )
        return Completion((findings, traces), usage)

    def run_agent(
        self,
        model: Any,
        email: InboundEmail,
        triage: Triage,
        case_id: str | None = None,
    ) -> tuple[Findings, list[ToolTrace]]:
        from langchain.agents import create_agent

        traces: list[ToolTrace] = []
        tools = make_tools(self._enterprise, self._store, email.sender, traces, case_id)
        agent = create_agent(model, tools, system_prompt=INVESTIGATE_SYSTEM, response_format=Findings)
        human = (
            f"Triage: category={triage.category}, severity={triage.severity}.\n\n"
            f"{wrap_untrusted(email, triage_summary=triage.summary)}"
        )
        result: dict[str, Any] = {}
        try:
            state: InputAgentState = {"messages": [HumanMessage(content=human)]}
            config: RunnableConfig = {"recursion_limit": 20}
            result = agent.invoke(state, config)
        except Exception as exc:  # noqa: BLE001  # logged; a capped run still keeps the lookups it finished
            log_failure(case_id or "-", "investigate", exc)
            if isinstance(exc, GraphRecursionError) and traces:
                result = {}
            else:
                raise InvestigationFailed(type(exc).__name__, self._fallback(traces), traces) from None
        self._attach_usage(result.get("messages") or [], traces)
        parsed = result.get("structured_response")
        if isinstance(parsed, Findings):
            findings = parsed
        elif isinstance(parsed, dict):
            try:
                findings = Findings.model_validate(parsed)
            except ValidationError as exc:
                log_failure(case_id or "-", "investigate", exc)
                findings = self._fallback(traces)
        else:
            findings = self._fallback(traces)
        return findings, traces

    def _fallback(self, traces: list[ToolTrace]) -> Findings:
        facts = _facts_from_traces(traces)
        nothing = _nothing_relevant(facts)
        if not traces:
            summary = "Investigation stopped before any lookup."
        elif len(traces) >= TOOL_CAP:
            summary = f"Stopped at the {TOOL_CAP} tool-call cap. " + (facts[0].text if facts else "No usable facts.")
        elif nothing:
            summary = "Nothing relevant found in service history, recalls, or earlier cases."
        else:
            summary = facts[0].text
        return Findings(summary=summary, facts=facts, nothing_found=nothing)

    def _attach_usage(self, messages: list[Any], traces: list[ToolTrace]) -> None:
        usages: list[dict[str, Any]] = []
        pending: dict[str, Any] = {}
        for message in messages:
            if isinstance(message, AIMessage) and message.usage_metadata:
                pending = dict(message.usage_metadata)
            if isinstance(message, ToolMessage) and message.name in TOOL_NAMES:
                usages.append(pending)
        for trace, usage in zip(traces, usages, strict=False):
            if usage.get("input_tokens") is not None:
                trace.input_tokens = int(usage["input_tokens"])
            if usage.get("output_tokens") is not None:
                trace.output_tokens = int(usage["output_tokens"])

    def _offline_run(
        self, email: InboundEmail, triage: Triage, case_id: str | None = None
    ) -> tuple[Findings, list[ToolTrace]]:
        traces: list[ToolTrace] = []

        def call(name: str, args: dict[str, Any], result: str) -> None:
            traces.append(ToolTrace(name, args, result))

        call("get_customer_profile", {}, profile_text(self._enterprise, email.sender))
        call("find_prior_cases", {}, prior_cases_text(self._store, email.sender, case_id))
        customer = self._enterprise.customer(email.sender)
        if triage.category in {"vehicle_fault", "warranty"} and customer and customer.vehicle:
            call(
                "get_service_history",
                {"vin": customer.vehicle.vin},
                history_text(self._enterprise, email.sender, customer.vehicle.vin),
            )
            call(
                "check_recalls",
                {},
                recalls_text(self._enterprise, customer.vehicle.model, customer.vehicle.model_year),
            )
        facts = _facts_from_traces(traces)
        nothing = _nothing_relevant(facts)
        if nothing:
            summary = "Nothing relevant found in service history, recalls, or earlier cases."
        else:
            summary = " ".join(fact.text for fact in facts if fact.source_tool != "get_customer_profile")
        return Findings(summary=summary, facts=facts, nothing_found=nothing), traces
