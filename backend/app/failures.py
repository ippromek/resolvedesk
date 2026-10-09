"""Safe failure logging.

A log line names the case, the step, the exception type and the file:line of the
raising frame. It never includes the exception message. Set LOG_TRACEBACKS=1 only
on a local machine when debugging; it prints the full traceback, which can contain
customer text. Leave it unset for the demo.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx
from anthropic import APIError as AnthropicError
from langchain_core.exceptions import OutputParserException
from openai import APIError as OpenAIError

PROVIDER_ERRORS: tuple[type[BaseException], ...] = (
    httpx.HTTPError,
    AnthropicError,
    OpenAIError,
    TimeoutError,
    OutputParserException,
)


logger = logging.getLogger(__name__)

MODEL_STEP_FAILURE = "The model did not respond. Try again."
OTHER_STEP_FAILURE = "This step failed. Try again."


def step_failure_message(exc: BaseException) -> str:
    """Provider and transport failures are a model outage. Everything else is a step failure."""
    if isinstance(exc, PROVIDER_ERRORS):
        return MODEL_STEP_FAILURE
    return OTHER_STEP_FAILURE


def _where(exc: BaseException) -> str:
    tb = exc.__traceback__
    frame = None
    while tb is not None:
        frame = tb.tb_frame
        tb = tb.tb_next
    if frame is None:
        return "unknown"
    return f"{Path(frame.f_code.co_filename).name}:{frame.f_lineno}"


def log_failure(case_id: str, step: str, exc: BaseException) -> None:
    kind = type(exc).__name__
    where = _where(exc)
    if os.getenv("LOG_TRACEBACKS") == "1":
        logger.exception("case %s failed at %s (%s at %s)", case_id, step, kind, where)
        return
    logger.error("case %s failed at %s (%s at %s)", case_id, step, kind, where)
