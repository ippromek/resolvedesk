"""Cheap, deterministic guards around the LLM steps.

These are a first line, not the defence. The real defences are structural:
the customer text is wrapped as data, the assistant has no write tools, and
nothing is sent without a human decision recorded by the graph.
"""

from __future__ import annotations

import re

from app.schemas import Draft, Grounding, GuardResult, InboundEmail, PolicySnippet

_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore (all |any )?(previous|prior|above) (instructions|rules)", "asks the model to ignore its instructions"),
    (r"\b(system prompt|developer message)\b", "mentions the system prompt"),
    (r"\byou are now\b|\bact as\b", "tries to change the model's role"),
    (
        r"(approve|send|issue).{0,40}(refund|compensation|voucher).{0,40}(immediately|now|automatically)",
        "demands an automatic payout",
    ),
    (r"(classif|mark|set).{0,30}(severity|priority).{0,30}(low|critical)", "tries to set its own severity"),
]

# Shared with the data wrappers so a spaced or attributed tag cannot close a block.
DATA_DELIMITER = re.compile(r"<\s*/?\s*(?:customer_email|tool_results|draft)\b[^>]*>", re.IGNORECASE)

# Commitments a draft must not make unless a cited policy states them.
_COMMITMENT = re.compile(
    r"(full refund|we will refund|compensation of|AED\s?\d[\d,]*|\d+\s?%|free of charge|free service|guarantee|voucher|discount|goodwill|complimentary)",
    re.IGNORECASE,
)
_AMOUNT = re.compile(r"^(?:AED\s?\d[\d,]*|\d+\s?%)$", re.IGNORECASE)
# A sentence is an offer only when it says one of these. Entries are matched
# against the lower-cased sentence.
_PROMISE_PHRASES: tuple[str, ...] = (
    "we will",
    "we'll",
    "you will receive",
    "we'll refund",
    "we can offer",
    "we are pleased to offer",
    "is approved",
    "we have applied",
    "here is your",
)
# A refusal is not a commitment, even when it names the thing being refused.
# Entries are regular expressions matched against the lower-cased clause.
_OFFER_NEGATIONS: tuple[str, ...] = (
    "unable to",
    "cannot",
    "can't",
    "not able to",
    "do not provide",
    r"no \w+ available",
    "not eligible",
)
# A later clause can still be an offer ("we cannot X, but we will refund").
_CLAUSE_BREAK = re.compile(r"[,;:]|\b(?:but|so|and|however)\b", re.IGNORECASE)
# The draft must not claim an action was taken. The arranged section is added later.
_ACTION_PROMISES: tuple[str, ...] = (
    "we are alerting",
    "we have alerted",
    "we will escalate",
    "we have escalated",
    "we have booked",
    "we have scheduled",
    "your reference is",
    "reference number",
)


def strip_delimiters(text: str) -> str:
    """Remove customer_email and tool_results tags, including spaced or attributed forms."""
    return DATA_DELIMITER.sub("", text)


def screen_input(email: InboundEmail) -> GuardResult:
    text = f"{email.subject}\n{email.body}"
    reasons = [label for pattern, label in _INJECTION_PATTERNS if re.search(pattern, text, re.IGNORECASE)]
    if DATA_DELIMITER.search(text):
        reasons.append("contains our data delimiter")
    return GuardResult(injection_suspected=bool(reasons), reasons=reasons)


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    previous = max(text.rfind(mark, 0, start) for mark in ".!?\n")
    begin = 0 if previous < 0 else previous + 1
    nxt = re.search(r"[.!?\n]", text[end:])
    stop = len(text) if nxt is None else end + nxt.start()
    return begin, stop


def _negated_before_match(sentence: str, match_start: int) -> bool:
    """True only when a refusal sits in the same clause, before the matched term."""
    start = 0
    for sep in _CLAUSE_BREAK.finditer(sentence):
        if sep.end() <= match_start:
            start = sep.end()
            continue
        break
    clause = sentence[start:match_start].lower()
    return any(re.search(pattern, clause) for pattern in _OFFER_NEGATIONS)


def _figure_in_email(phrase: str, email_body: str) -> bool:
    compact = re.sub(r"[\s,]", "", phrase).lower()
    if not compact:
        return False
    return compact in re.sub(r"[\s,]", "", email_body).lower()


def check_grounding(draft: Draft, snippets: list[PolicySnippet], email_body: str = "") -> Grounding:
    issues: list[str] = []
    known = {s.id for s in snippets}
    unknown = [c for c in draft.cited_policy_ids if c not in known]
    if unknown:
        issues.append(f"cites policies that were not retrieved: {', '.join(unknown)}")
    if snippets and not draft.cited_policy_ids:
        issues.append("cites no policy")
    cited_text = " ".join(s.text for s in snippets if s.id in draft.cited_policy_ids).lower()
    for match in _COMMITMENT.finditer(draft.body):
        phrase = match.group(0)
        begin, _stop = _sentence_bounds(draft.body, match.start(), match.end())
        sentence = draft.body[begin:_stop]
        relative = match.start() - begin
        if _negated_before_match(sentence, relative):
            continue
        if phrase.lower() in cited_text:
            continue
        if (
            _AMOUNT.match(phrase)
            and _figure_in_email(phrase, email_body)
            and not any(promise in sentence.lower() for promise in _PROMISE_PHRASES)
        ):
            continue
        issues.append(f"makes a commitment not found in cited policy: '{phrase}'")
    lowered_body = draft.body.lower()
    for phrase in _ACTION_PROMISES:
        if phrase in lowered_body:
            issues.append(f"states an action that has not been approved: '{phrase}'")
    if len(draft.body.split()) > 260:
        issues.append("reply is longer than 260 words")
    return Grounding(ok=not issues, issues=issues)
