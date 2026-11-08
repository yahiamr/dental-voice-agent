"""Document search behind one interface. Stage: 1 and 5.

What this file is:
    Ways of searching the clinic documents, all called the same way.

What it contains:
    - Retriever: search(question) -> up to 3 SearchResult, and rank(question, k) -> the top k.
    - NoRetriever: no index at all, so every search raises SearchUnavailable.
    - KeywordRetriever(path): SQLite full-text search, ranked by BM25.
    - VectorRetriever(path, embedder): the nearest pieces by meaning, from sqlite-vec.
    - HybridRetriever(path, embedder): both rankings, merged by reciprocal rank fusion, or the
      keyword ranking alone when vector search fails.
    - fusion_scores(rankings) and reciprocal_rank_fusion(rankings): the merge.
    - keyword_query(question): turns a question into a safe full-text query.

How to understand it:
    Keyword search finds shared words, and is exact about names such as "Bizum" or "SER".
    Vector search finds similar meaning, so "somewhere to leave my car" can find "By car" with
    no word in common. Hybrid search takes the top 5 of each and merges them by position:
    reciprocal rank fusion gives each piece 1 / (60 + its rank) in each list and adds them up,
    so a piece both searches like rises, and one search alone can still bring a piece in. It
    needs no tuning, which suits an index of about 50 pieces. 3 pieces are returned.

    keyword_query drops filler words ("can", "the", "my") and quotes every remaining word,
    joined with OR. Quoting means caller text such as OR, NEAR, * or a stray quote can never act
    as full-text syntax. OR means any word may match, and BM25 puts pieces that match more of
    them first. A word in a heading counts double, because a heading says what a piece is about.

    A search never decides whether its results answer the question: keyword search returns any
    piece sharing a word, and vector search always returns its nearest pieces. The prompt tells
    the model to answer only from what the results say, and otherwise to give the clinic number.

    Each search opens and closes its own connection, because LangGraph may run tools on other
    threads. Booking never depends on search: any database or embedding problem raises
    SearchUnavailable, and the agent gives the clinic number. Hybrid search first falls back to
    keywords, so only a keyword failure reaches the caller.

Uses:
    clinic/index.py, models.py

Used by:
    factories.py, tools/read_tools.py, evals/retrieval_eval.py
"""

import logging
import re
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from typing import Protocol

from dental_agent.clinic.index import Embedder, connect
from dental_agent.models import SearchResult

RESULTS = 3  # Pieces returned to the model.
CANDIDATES = 5  # Pieces each ranking offers to the merge.
RRF_K = 60  # The usual constant for reciprocal rank fusion.
HEADING_WEIGHT = 2.0  # A word in a heading counts this many times a word in the text.

# Words that say nothing about which piece answers a question, kept as one string because 80
# quoted words would be harder to read.
_FILLER_TEXT = """
a about after all also am an and any are as at be been before but by can could do does for from
get got had has have how i if in into is it its just me might more my no not of on one or our
please should so some than that the their them then there these they this to too us want was we
were what when where which who why will with would yes you your
"""
FILLER_WORDS = frozenset(_FILLER_TEXT.split())

logger = logging.getLogger(__name__)


class SearchUnavailable(Exception):
    """Search cannot run, so the agent gives the clinic number instead."""


class Retriever(Protocol):
    """What every search backend must offer."""

    def search(self, question: str) -> list[SearchResult]:
        """Return up to 3 document pieces that may answer the question."""
        ...


class NoRetriever:
    """No index is available, so every search refuses."""

    def search(self, question: str) -> list[SearchResult]:
        """Always refuse, so the search tool returns search_unavailable."""
        raise SearchUnavailable("No search index is available.")


def keyword_query(question: str) -> str | None:
    """Return a full-text query matching any meaningful word, or None if there is none."""
    words = [word for word in re.findall(r"\w+", question.lower()) if word not in FILLER_WORDS]
    unique = list(dict.fromkeys(words))  # Keeps the order and drops repeats.
    return " OR ".join(f'"{word}"' for word in unique) or None


def fusion_scores(rankings: Iterable[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Return each piece id's reciprocal rank fusion score, best first."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, piece_id in enumerate(ranking, start=1):
            scores[piece_id] = scores.get(piece_id, 0.0) + 1.0 / (k + position)
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


def reciprocal_rank_fusion(rankings: Iterable[list[str]], k: int = RRF_K) -> list[str]:
    """Merge rankings of piece ids into one, best first."""
    return list(fusion_scores(rankings, k))


class _IndexSearch:
    """What every search over the index shares: opening it, and turning rows into results."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def search(self, question: str) -> list[SearchResult]:
        """Return the best 3 pieces for the question."""
        return self.rank(question, RESULTS)

    def rank(self, question: str, k: int) -> list[SearchResult]:
        """Return the best k pieces, best first."""
        if not self.path.is_file():
            raise SearchUnavailable(f"No search index at {self.path}.")
        try:
            # closing, because a connection's own with block commits but never closes it.
            with closing(connect(self.path)) as connection:
                connection.row_factory = sqlite3.Row
                return self._rank(connection, question, k)
        except sqlite3.Error as error:
            raise SearchUnavailable(str(error)) from error

    def _rank(self, connection: sqlite3.Connection, question: str, k: int) -> list[SearchResult]:
        raise NotImplementedError


def _result(row: sqlite3.Row, score: float) -> SearchResult:
    """Turn a chunks row into a SearchResult."""
    return SearchResult(
        text=row["text"],
        source=row["source"],
        score=score,
        heading=row["heading"],
        piece_id=row["piece_id"],
    )


class KeywordRetriever(_IndexSearch):
    """Full-text search over headings and text, ranked by BM25."""

    def _rank(self, connection: sqlite3.Connection, question: str, k: int) -> list[SearchResult]:
        query = keyword_query(question)
        if query is None:
            return []
        rows = connection.execute(
            """
            SELECT chunks.*, bm25(chunks_fts, ?, 1.0) AS rank
            FROM chunks_fts JOIN chunks ON chunks.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY rank LIMIT ?
            """,
            (HEADING_WEIGHT, query, k),
        ).fetchall()
        # BM25 is lower for a better match, so it is negated: higher is better everywhere.
        return [_result(row, -row["rank"]) for row in rows]


class VectorRetriever(_IndexSearch):
    """The pieces whose embeddings are nearest the question's."""

    def __init__(self, path: Path, embedder: Embedder) -> None:
        super().__init__(path)
        self.embedder = embedder

    def _rank(self, connection: sqlite3.Connection, question: str, k: int) -> list[SearchResult]:
        import sqlite_vec

        try:
            vector = sqlite_vec.serialize_float32(self.embedder.embed_query(question))
        except Exception as error:  # A model runtime failure is a search failure too.
            raise SearchUnavailable(f"The question could not be embedded: {error}") from error
        try:
            rows = connection.execute(
                """
                SELECT chunks.*, nearest.distance AS distance
                FROM (
                  SELECT rowid, distance FROM chunks_vec WHERE embedding MATCH ? AND k = ?
                ) AS nearest
                JOIN chunks ON chunks.id = nearest.rowid
                ORDER BY nearest.distance
                """,
                (vector, k),
            ).fetchall()
        except sqlite3.OperationalError as error:
            # An index built without an embedder has no vector table.
            raise SearchUnavailable(f"This index has no vectors: {error}") from error
        # A smaller distance is a better match, so it is negated: higher is better everywhere.
        return [_result(row, -row["distance"]) for row in rows]


class HybridRetriever:
    """Keyword and vector search, merged by reciprocal rank fusion."""

    def __init__(self, path: Path, embedder: Embedder) -> None:
        self.keyword = KeywordRetriever(path)
        self.vector = VectorRetriever(path, embedder)

    def search(self, question: str) -> list[SearchResult]:
        """Return the best 3 pieces for the question."""
        return self.rank(question, RESULTS)

    def rank(self, question: str, k: int) -> list[SearchResult]:
        """Return the best k pieces of the merged rankings, with their fusion score.

        If vector search fails, such as on an index rebuilt without vectors, the keyword
        ranking is used alone, so a caller still gets an answer.
        """
        lists = [self.keyword.rank(question, CANDIDATES)]
        try:
            lists.append(self.vector.rank(question, CANDIDATES))
        except SearchUnavailable as error:
            logger.warning("Vector search failed, so this search uses keywords only: %s", error)
        found = {item.piece_id: item for ranking in lists for item in ranking}
        scores = fusion_scores([item.piece_id for item in ranking] for ranking in lists)
        return [
            found[piece_id].model_copy(update={"score": score})
            for piece_id, score in list(scores.items())[:k]
        ]
