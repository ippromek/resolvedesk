"""A deliberately small knowledge base: markdown policies split by `## ` heading.

Each section's id is written in its heading, e.g. `## [WAR-1] Standard warranty`.
Retrieval is BM25 over those sections, boosted for the triaged category. That is
enough for a dozen policies. Swapping in a vector store changes `search` only.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.schemas import PolicySnippet

_HEADING = re.compile(r"^##\s*\[(?P<id>[A-Z]+-\d+)\]\s*(?P<title>.+)$", re.MULTILINE)
_CATEGORY = re.compile(r"^categories:\s*(.+)$", re.MULTILINE)
_TOKEN = re.compile(r"[a-z؀-ۿ]{3,}")
_CATEGORY_BOOST = 3.0
# A strong hit is at least three times the category boost, so one on-topic
# policy plus weaker neighbours still counts as thin retrieval.
_STRONG_SCORE = _CATEGORY_BOOST * 3
_FALLBACK_ID = "CMP-1"


@dataclass(frozen=True)
class Section:
    id: str
    title: str
    text: str
    categories: tuple[str, ...]


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def load_sections(kb_dir: Path) -> list[Section]:
    sections: list[Section] = []
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        cat_match = _CATEGORY.search(raw)
        cats = tuple(c.strip() for c in cat_match.group(1).split(",")) if cat_match else ()
        heads = list(_HEADING.finditer(raw))
        for i, head in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(raw)
            body = raw[head.end() : end].strip()
            sections.append(Section(head.group("id"), head.group("title").strip(), body, cats))
    return sections


class KnowledgeBase:
    def __init__(self, sections: list[Section]) -> None:
        self.sections = sections
        self._docs = [_tokens(f"{s.title} {s.text}") for s in sections]
        self._avg = sum(len(d) for d in self._docs) / max(len(self._docs), 1)
        df: Counter[str] = Counter()
        for d in self._docs:
            df.update(set(d))
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def get(self, section_id: str) -> Section | None:
        return next((s for s in self.sections if s.id == section_id), None)

    def search(self, query: str, category: str | None = None, k: int = 3) -> list[PolicySnippet]:
        q = _tokens(query)
        scored: list[tuple[float, Section]] = []
        for section, doc in zip(self.sections, self._docs, strict=True):
            tf = Counter(doc)
            score = 0.0
            for term in q:
                if term not in tf:
                    continue
                f = tf[term]
                score += self._idf[term] * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * len(doc) / self._avg))
            if category and category in section.categories:
                score += _CATEGORY_BOOST
            if score > 0:
                scored.append((score, section))
        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = scored[:k]
        strong = sum(1 for sc, _ in chosen if sc > _STRONG_SCORE)
        if strong < 2:
            fallback = next((pair for pair in scored if pair[1].id == _FALLBACK_ID), None)
            if fallback is None:
                fallback_section = self.get(_FALLBACK_ID)
                if fallback_section is not None:
                    fallback = (0.0, fallback_section)
            if fallback is not None and all(item.id != _FALLBACK_ID for _, item in chosen):
                chosen = [*chosen, fallback]
        return [PolicySnippet(id=s.id, title=s.title, text=s.text, score=round(sc, 2)) for sc, s in chosen]


@lru_cache(maxsize=4)
def get_kb(kb_dir: Path) -> KnowledgeBase:
    return KnowledgeBase(load_sections(kb_dir))
