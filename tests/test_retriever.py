"""Tests for document search. Stage: 5.

What this file is:
    Proves keyword, vector and hybrid search over a small index, and each failure case.

What it contains:
    - keyword_query: filler words dropped, words quoted, so caller text can never become FTS
      syntax, and nothing left means no search.
    - Keyword search: finds by shared words, with stemming, ranks a heading match above a
      body match, and returns nothing when no word matches.
    - Vector search: finds by meaning with FakeEmbedder.
    - reciprocal_rank_fusion, and hybrid search finding a piece only one ranking found.
    - At most 3 results, each with its heading, text, source and piece id.
    - A missing index, or vector search on an index without vectors, raises SearchUnavailable.

How to understand it:
    The index is built from a small sample with FakeEmbedder (tests/fakes.py), so no model is
    downloaded. The real handbook and the real model are measured in evals/retrieval_eval.py.

Used by:
    pytest
"""

import pytest
from tests.fakes import FakeEmbedder

from dental_agent.clinic.index import build_index
from dental_agent.clinic.retriever import (
    HybridRetriever,
    KeywordRetriever,
    SearchUnavailable,
    VectorRetriever,
    keyword_query,
    reciprocal_rank_fusion,
)

SAMPLE = """# Handbook

## Finding us

### By car

There is no parking at the clinic. The public car park is 150 metres away.

### By metro

The nearest metro station is Iglesia.

## Changing an appointment

You can cancel or move an appointment with 24 hours' notice.

## Hygiene

Instruments are cleaned and sterilised between patients. Parking attendants are not staff.

## Prices

A cleaning costs 60 euros.
"""


@pytest.fixture
def rag_path(tmp_path):
    docs = tmp_path / "handbook.md"
    docs.write_text(SAMPLE)
    path = tmp_path / "rag.sqlite"
    build_index([docs], path, FakeEmbedder())
    return path


def headings(results):
    return [item.heading for item in results]


# The keyword query


def test_keyword_query_drops_filler_words_and_quotes_the_rest():
    assert keyword_query("Where can I park my car?") == '"park" OR "car"'


def test_caller_text_can_never_become_search_syntax():
    # Quotes, OR, NEAR and stars are FTS operators. Quoting each word makes them plain words.
    # "or" itself is a filler word, so it is dropped like the others.
    assert keyword_query('car" OR NEAR(* -x') == '"car" OR "near" OR "x"'


def test_a_question_of_only_filler_words_gives_no_query():
    assert keyword_query("Can you do it?") is None


# Keyword search


def test_keyword_search_finds_the_piece_that_shares_words(rag_path):
    assert headings(KeywordRetriever(rag_path).search("car park"))[0] == "Finding us › By car"


def test_keyword_search_uses_stemming(rag_path):
    results = KeywordRetriever(rag_path).search("cancelling")
    assert headings(results) == ["Changing an appointment"]


def test_a_heading_match_ranks_above_a_body_match(rag_path):
    # "Hygiene" mentions parking in passing. The By car piece is about it.
    results = KeywordRetriever(rag_path).search("parking")
    assert headings(results)[0] == "Finding us › By car"


def test_keyword_search_returns_nothing_when_no_word_matches(rag_path):
    assert KeywordRetriever(rag_path).search("dragons unicorns") == []


# Vector search and merging


def test_vector_search_finds_by_meaning(rag_path):
    results = VectorRetriever(rag_path, FakeEmbedder()).search("somewhere to leave the car")
    assert headings(results)[0] == "Finding us › By car"


def test_reciprocal_rank_fusion_rewards_a_piece_both_rankings_like():
    merged = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]])
    assert merged[:2] == ["b", "a"] or merged[:2] == ["a", "b"]
    assert set(merged) == {"a", "b", "c", "d"}
    # b is 2nd and 1st, a is 1st and 3rd: b scores 1/62 + 1/61, a scores 1/61 + 1/63.
    assert merged[0] == "b"


def test_hybrid_search_returns_a_piece_only_the_vector_ranking_found(rag_path):
    # No word of this question is in the metro piece, but its meaning matches.
    results = HybridRetriever(rag_path, FakeEmbedder()).search("metro")
    assert "Finding us › By metro" in headings(results)


def test_search_returns_at_most_three_results_with_their_details(rag_path):
    results = HybridRetriever(rag_path, FakeEmbedder()).search("car parking metro euros cleaning")
    assert len(results) == 3
    first = results[0]
    assert first.piece_id and first.heading and first.text
    assert first.source == "handbook.md"


# Failures


def test_a_missing_index_raises_search_unavailable(tmp_path):
    with pytest.raises(SearchUnavailable):
        KeywordRetriever(tmp_path / "missing.sqlite").search("car")


def test_vector_search_on_an_index_without_vectors_raises_search_unavailable(tmp_path):
    docs = tmp_path / "handbook.md"
    docs.write_text(SAMPLE)
    path = tmp_path / "rag.sqlite"
    build_index([docs], path, None)
    with pytest.raises(SearchUnavailable):
        VectorRetriever(path, FakeEmbedder()).search("car")
    assert headings(KeywordRetriever(path).search("car"))[0] == "Finding us › By car"


class BrokenEmbedder(FakeEmbedder):
    """Embeds the documents, then fails on every question, as a broken model runtime might."""

    def embed_query(self, text):
        raise RuntimeError("model runtime failed")


def test_an_embedding_failure_raises_search_unavailable(rag_path):
    with pytest.raises(SearchUnavailable):
        VectorRetriever(rag_path, BrokenEmbedder()).search("car")


def test_hybrid_search_uses_keywords_on_an_index_without_vectors(tmp_path):
    docs = tmp_path / "handbook.md"
    docs.write_text(SAMPLE)
    path = tmp_path / "rag.sqlite"
    build_index([docs], path, None)
    assert headings(HybridRetriever(path, FakeEmbedder()).search("car"))[0] == "Finding us › By car"


def test_hybrid_search_uses_keywords_when_the_model_fails(rag_path):
    results = HybridRetriever(rag_path, BrokenEmbedder()).search("car")
    assert headings(results)[0] == "Finding us › By car"
