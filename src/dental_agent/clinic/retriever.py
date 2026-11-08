"""Document search behind one interface. Stage: 1 and 5.

What this file is:
    Three ways of searching the clinic documents, all called the same way.

What it contains:
    - Retriever.search(question) -> list[SearchResult].
    - NoRetriever: the Stage 1 placeholder, which raises SearchUnavailable.
    - KeywordRetriever(path): FTS5 with BM25 ranking.
    - HybridRetriever(path, embeddings): adds vector search and merges both rankings.

How to understand it:
    Keyword search finds shared words. Vector search finds similar meaning, so "leave my car"
    can match parking. Reciprocal rank fusion merges them by position in each list, with k=60.
    Booking never depends on search. A failure returns search_unavailable, and the agent gives
    the clinic number.

Uses:
    models.py

Used by:
    factories.py, tools/read_tools.py
"""

from typing import Protocol

from dental_agent.models import SearchResult


class SearchUnavailable(Exception):
    """Search cannot run, so the agent gives the clinic number instead."""


class Retriever(Protocol):
    """What every search backend must offer."""

    def search(self, question: str) -> list[SearchResult]:
        """Return up to 3 document sections that may answer the question."""
        ...


class NoRetriever:
    """The Stage 1 placeholder. Keyword and hybrid search arrive in Stage 5."""

    def search(self, question: str) -> list[SearchResult]:
        """Always refuse, so the search tool returns search_unavailable."""
        raise SearchUnavailable("No search index is available yet.")
