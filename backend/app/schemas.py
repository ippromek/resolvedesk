"""Typed contracts shared by the graph, the API and the eval.

Every LLM output is parsed into one of these models, so a malformed answer
fails loudly at the boundary instead of leaking into the case record.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Category = Literal[
    "vehicle_fault",
    "service_experience",
    "warranty",
    "billing",
    "sales",
    "delivery_delay",
    "data_privacy",
    "spam",
    "other",
]
Severity = Literal["low", "medium", "high", "critical"]
Language = Literal["en", "ar"]
Decision = Literal["approve", "edit", "reject"]
CaseStatus = Literal[
    "processing",
    "awaiting_review",
    "sent",
    "rejected",
    "closed_no_reply",
    "failed",
]

CATEGORIES: tuple[str, ...] = Category.__args__  # type: ignore[attr-defined]
SEVERITIES: tuple[str, ...] = Severity.__args__  # type: ignore[attr-defined]


class InboundEmail(BaseModel):
    sender: str = Field(min_length=3, max_length=200)
    subject: str = Field(min_length=1, max_length=300)
    body: str = Field(min_length=1, max_length=20_000)


class GuardResult(BaseModel):
    injection_suspected: bool
    reasons: list[str] = Field(default_factory=list)


class Triage(BaseModel):
    """What the triage step must return."""

    category: Category
    severity: Severity = Field(description="Follows risk to the customer, not tone or anger.")
    language: Language
    summary: str = Field(description="One sentence, neutral, no customer PII.")
    rationale: str = Field(description="Why this category and severity, citing the email.")


class PolicySnippet(BaseModel):
    id: str
    title: str
    text: str
    score: float


class Draft(BaseModel):
    """What the drafting step must return."""

    body: str = Field(description="The reply to the customer, in the customer's language.")
    cited_policy_ids: list[str] = Field(
        default_factory=list, description="IDs of the policy snippets the reply relies on."
    )


class Fact(BaseModel):
    text: str
    source_tool: str


class Findings(BaseModel):
    """What the investigator must return. Every fact comes from a tool result."""

    summary: str
    facts: list[Fact] = Field(default_factory=list)
    nothing_found: bool = False


ActionType = Literal[
    "schedule_roadside_pickup",
    "book_safety_inspection",
    "register_recall_repair",
    "open_refund_review",
    "escalate_to_branch_manager",
    "offer_courtesy_car",
    "forward_to_data_protection",
]
ACTION_TYPES: tuple[str, ...] = ActionType.__args__  # type: ignore[attr-defined]


class ProposedAction(BaseModel):
    """One action the model wants a reviewer to approve. Ids are assigned after validation."""

    type: str
    args: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)


class ActionIdea(BaseModel):
    """Flat schema for the model. Nested dicts come back empty on this provider."""

    type: str
    branch: str = ""
    priority: str = ""
    recall_id: str = ""
    invoice_ref: str = ""
    reason: str = ""
    request_type: str = ""
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)


class ActionBatch(BaseModel):
    proposals: list[ActionIdea] = Field(default_factory=list)


class Grounding(BaseModel):
    ok: bool
    issues: list[str] = Field(default_factory=list)


class ReviewInput(BaseModel):
    decision: Decision
    reviewer: str = Field(min_length=1, max_length=100)
    body: str | None = Field(default=None, description="Required when decision is 'edit'.")
    note: str | None = Field(default=None, max_length=1000)
    # Empty by default so an older client approves the reply and no actions.
    approved_action_ids: list[str] = Field(default_factory=list)
    # Required for approve and edit when the guard flagged the email. Reject does not need it.
    injection_acknowledged: bool = False


class PreviewInput(BaseModel):
    approved_action_ids: list[str] = Field(default_factory=list)
    body: str | None = Field(default=None, description="Edited reply text. The draft is used when omitted.")


class AssistantQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class AssistantAnswer(BaseModel):
    answer: str
    tools_used: list[str]
