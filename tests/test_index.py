"""Tests for the search index. Stage: 5.

What this file is:
    Proves how the handbook is split into pieces, and how the index is built and kept current.

What it contains:
    - split_sections: one piece per ### subsection, or per ## section without subsections, the
      heading path on every piece, tables kept whole, the header comment left out, unique ids.
    - The real handbook splits into about 50 pieces, with no empty piece.
    - build_index, index_is_current and ensure_index: a fresh build, the fingerprint, and a
      rebuild when the document or the embedding model changes.

How to understand it:
    FakeEmbedder, from tests/fakes.py, stands in for FastEmbed, so these tests need no model
    download.

Used by:
    pytest
"""

import sqlite3
from pathlib import Path

import pytest
from tests.fakes import FakeEmbedder

from dental_agent.clinic.index import (
    build_index,
    ensure_index,
    index_is_current,
    split_sections,
)

HANDBOOK = Path(__file__).parent.parent / "data" / "docs" / "patient-handbook.md"

SAMPLE = """<!--
A header comment that must never be indexed.
-->

# Sample Handbook

Welcome text before any section.

## Finding us

The clinic is on the first floor.

### By car

There is no parking at the clinic.

### By metro

The nearest station is Iglesia.

## Prices

| Treatment | Price |
| --- | --- |
| Cleaning | 60 euros |

## Questions

### Can I bring my dog?

Only assistance dogs.
"""


# Splitting


def test_each_subsection_is_one_piece_with_its_heading_path():
    pieces = split_sections(SAMPLE, "sample.md")
    assert [piece.heading for piece in pieces] == [
        "Sample Handbook",
        "Finding us",
        "Finding us › By car",
        "Finding us › By metro",
        "Prices",
        "Questions › Can I bring my dog?",
    ]
    by_car = pieces[2]
    assert by_car.text == "There is no parking at the clinic."
    assert by_car.source == "sample.md"
    assert by_car.piece_id == "finding-us/by-car"


def test_a_section_with_no_text_before_its_first_subsection_gives_no_empty_piece():
    headings = [piece.heading for piece in split_sections(SAMPLE, "sample.md")]
    assert "Questions" not in headings


def test_a_table_is_kept_whole_inside_its_section():
    prices = next(item for item in split_sections(SAMPLE, "s.md") if item.heading == "Prices")
    assert "| Cleaning | 60 euros |" in prices.text
    assert prices.text.startswith("| Treatment | Price |")


def test_the_header_comment_is_never_indexed():
    assert all("header comment" not in piece.text for piece in split_sections(SAMPLE, "s.md"))


def test_the_real_handbook_splits_into_about_fifty_pieces():
    pieces = split_sections(HANDBOOK.read_text(), HANDBOOK.name)
    assert 45 <= len(pieces) <= 55
    assert all(piece.text.strip() for piece in pieces)
    assert len({piece.piece_id for piece in pieces}) == len(pieces)
    assert "Finding us › By car" in [piece.heading for piece in pieces]


# Building and keeping the index current


@pytest.fixture
def docs(tmp_path):
    path = tmp_path / "handbook.md"
    path.write_text(SAMPLE)
    return [path]


def rows(rag_path, table):
    with sqlite3.connect(rag_path) as connection:
        return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_build_index_stores_every_piece_with_its_vectors(tmp_path, docs):
    rag_path = tmp_path / "rag.sqlite"
    count = build_index(docs, rag_path, FakeEmbedder())
    assert count == 6
    assert rows(rag_path, "chunks") == 6
    assert index_is_current(rag_path, docs, "fake-embedder")


def test_a_changed_document_makes_the_index_out_of_date(tmp_path, docs):
    rag_path = tmp_path / "rag.sqlite"
    build_index(docs, rag_path, FakeEmbedder())
    docs[0].write_text(SAMPLE + "\n## New section\n\nNew text.\n")
    assert not index_is_current(rag_path, docs, "fake-embedder")


def test_another_embedding_model_makes_the_index_out_of_date(tmp_path, docs):
    rag_path = tmp_path / "rag.sqlite"
    build_index(docs, rag_path, FakeEmbedder())
    assert not index_is_current(rag_path, docs, "another-model")


def test_a_missing_index_is_out_of_date(tmp_path, docs):
    assert not index_is_current(tmp_path / "missing.sqlite", docs, "fake-embedder")


def test_ensure_index_builds_only_when_needed(tmp_path, docs):
    rag_path = tmp_path / "rag.sqlite"
    assert ensure_index(docs, rag_path, FakeEmbedder()) is True
    assert ensure_index(docs, rag_path, FakeEmbedder()) is False
    docs[0].write_text(SAMPLE.replace("Iglesia", "Quevedo"))
    assert ensure_index(docs, rag_path, FakeEmbedder()) is True


def test_a_rebuild_replaces_the_old_index_completely(tmp_path, docs):
    rag_path = tmp_path / "rag.sqlite"
    build_index(docs, rag_path, FakeEmbedder())
    docs[0].write_text("# Short\n\n## Only\n\nOne piece.\n")
    build_index(docs, rag_path, FakeEmbedder())
    assert rows(rag_path, "chunks") == 1


def test_a_keyword_only_check_accepts_a_current_index_with_vectors(tmp_path, docs):
    rag_path = tmp_path / "rag.sqlite"
    build_index(docs, rag_path, FakeEmbedder())
    # Keyword search works on it as it is, so it is not rebuilt without its vectors.
    assert ensure_index(docs, rag_path, None) is False
    assert rows(rag_path, "chunks_vec_rowids") == 6  # sqlite-vec's own table of the vectors.


def test_the_same_heading_in_two_documents_gets_two_ids(tmp_path, docs):
    other = tmp_path / "other.md"
    other.write_text("# Other\n\n## Finding us\n\n### By car\n\nAnother text.\n")
    rag_path = tmp_path / "rag.sqlite"
    assert build_index([*docs, other], rag_path, FakeEmbedder()) == 7
    with sqlite3.connect(rag_path) as connection:
        ids = [row[0] for row in connection.execute("SELECT piece_id FROM chunks")]
    assert len(set(ids)) == 7
    # The first document keeps its ids, so labelled questions stay valid.
    assert "finding-us/by-car" in ids
