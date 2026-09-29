"""``search_knowledge_base``: BM25 over the markdown runbooks in ``data/kb`` (PLAN §7.1).

Uses the BM25L variant. Classic BM25Okapi gives a zero or negative IDF to any term
found in half the documents or more, which on a corpus of eight documents means
queries like "severity matrix" score nothing at all.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from rank_bm25 import BM25L

from harness.tools.spec import RetryPolicy, ToolContext, ToolSpec

SNIPPET_MAX_CHARS = 500

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Common English words that would otherwise give weak matches to every document.
# fmt: off
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from", "how",
    "i", "in", "is", "it", "of", "on", "or", "our", "should", "the", "this", "to", "we",
    "what", "when", "which", "with",
})
# fmt: on


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class KBDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    title: str
    services: list[str]
    tags: list[str]
    body: str


def parse_document(text: str, source: str) -> KBDocument:
    """Parse a KB file: ``---`` front matter, then the markdown body.

    Front matter lines are ``key: value``, ``key: "quoted value"`` or ``key: [a, b]``.

    Only this flat subset of YAML is supported; anything else fails loudly so a
    malformed document is caught at startup rather than silently mis-indexed.
    """
    match = re.match(r"\A---\n(.*?)\n---\n(.*)\Z", text.replace("\r\n", "\n"), re.DOTALL)
    if match is None:
        raise ValueError(f"{source}: missing '---' front matter")
    header, body = match.groups()
    fields: dict[str, Any] = {}
    for line in header.splitlines():
        key, sep, value = line.partition(":")
        if not sep or not key.strip():
            raise ValueError(f"{source}: unsupported front matter line {line!r}")
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            fields[key.strip()] = [v.strip() for v in value[1:-1].split(",") if v.strip()]
        elif len(value) >= 2 and value[0] == value[-1] == '"':
            fields[key.strip()] = value[1:-1]  # quoted, e.g. a title containing ':'
        else:
            fields[key.strip()] = value
    return KBDocument(
        doc_id=fields.get("id", ""),
        title=fields.get("title", ""),
        services=fields.get("services", []),
        tags=fields.get("tags", []),
        body=body.strip(),
    )


class KBHit(BaseModel):
    doc_id: str
    title: str
    snippet: str = Field(max_length=SNIPPET_MAX_CHARS)
    score: float


class KnowledgeBase:
    """In-memory BM25 index, built once at startup."""

    def __init__(self, documents: Sequence[KBDocument]) -> None:
        if not documents:
            raise ValueError("knowledge base is empty")
        self._documents = list(documents)
        corpus = [tokenize(f"{d.title}\n{d.body}") for d in self._documents]
        self._index = BM25L(corpus)
        self._vocab = [set(tokens) for tokens in corpus]

    @classmethod
    def from_dir(cls, kb_dir: Path) -> KnowledgeBase:
        paths = sorted(kb_dir.glob("*.md"))
        return cls([parse_document(p.read_text(encoding="utf-8"), p.name) for p in paths])

    def search(self, query: str, top_k: int) -> list[KBHit]:
        terms = tokenize(query)
        if not terms:
            return []
        scores = self._index.get_scores(terms)
        # Only documents sharing at least one query term count as a match.
        matches = [
            (float(score), doc)
            for score, doc, vocab in zip(scores, self._documents, self._vocab, strict=True)
            if vocab.intersection(terms)
        ]
        matches.sort(key=lambda m: -m[0])
        return [
            KBHit(
                doc_id=doc.doc_id,
                title=doc.title,
                snippet=_best_snippet(doc.body, set(terms)),
                score=round(score, 3),
            )
            for score, doc in matches[:top_k]
        ]


def _best_snippet(body: str, terms: set[str]) -> str:
    """Text from the paragraph sharing the most query terms onwards, capped in length.

    Headings are skipped; the earliest paragraph wins ties. Continuing past the
    best paragraph keeps the context that usually follows a matching lead-in.
    """
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip() and not p.startswith("#")]
    if not paragraphs:
        paragraphs = [body]
    best = max(range(len(paragraphs)), key=lambda i: len(terms & set(tokenize(paragraphs[i]))))
    text = "\n\n".join(paragraphs[best:])
    if len(text) <= SNIPPET_MAX_CHARS:
        return text
    return text[: SNIPPET_MAX_CHARS - 1] + "…"


class SearchKnowledgeBaseInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=500, description="Free-text search query.")
    top_k: int = Field(default=3, ge=1, le=10, description="Maximum number of results.")


class SearchKnowledgeBaseOutput(BaseModel):
    results: list[KBHit]


def make_search_knowledge_base_tool(kb: KnowledgeBase, retry: RetryPolicy) -> ToolSpec:
    async def handler(args: SearchKnowledgeBaseInput, ctx: ToolContext) -> dict[str, Any]:
        hits = kb.search(args.query, args.top_k)
        return {"results": [h.model_dump() for h in hits]}

    return ToolSpec(
        name="search_knowledge_base",
        description=(
            "Search the operations knowledge base (runbooks, severity matrix, incident "
            "guidelines, maintenance windows). Returns the best-matching documents with a snippet."
        ),
        input_model=SearchKnowledgeBaseInput,
        output_model=SearchKnowledgeBaseOutput,
        handler=handler,
        timeout_s=3.0,
        retry=retry,
    )
