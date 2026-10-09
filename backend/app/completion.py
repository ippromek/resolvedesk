"""A model call's value and its own timing, returned together so concurrent cases cannot swap them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Usage:
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None

    def as_mapping(self) -> dict[str, int]:
        found: dict[str, int] = {"latency_ms": self.latency_ms}
        if self.input_tokens is not None:
            found["input_tokens"] = self.input_tokens
        if self.output_tokens is not None:
            found["output_tokens"] = self.output_tokens
        return found


@dataclass(frozen=True)
class Completion(Generic[T]):
    value: T
    usage: Usage
