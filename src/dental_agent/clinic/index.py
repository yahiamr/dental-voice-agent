"""Builds the search index. Stage: 5.

What this file is:
    Turns the Markdown documents in data/docs/ into one searchable SQLite database.

What it contains:
    - Piece: one searchable part of a document, with its id, heading path, text and source.
    - Embedder: what the index needs from an embedding model (FastEmbedder in clinic/embedder.py).
    - split_sections(markdown, source, seen): the pieces of one document.
    - build_index(docs, rag_path, embedder): writes the whole index, and returns the piece count.
    - index_is_current(rag_path, docs, model_name) and ensure_index(docs, rag_path, embedder).
    - connect(path): opens an index with the sqlite-vec extension loaded.

How to understand it:
    A piece is one ### subsection, or one ## section that has no subsections, so each piece
    answers one thing, such as "By car" or "After a filling". Text before the first subsection
    of a section is a piece of its own, and an empty one is skipped. Every piece carries its
    heading path, such as "Finding us › By car", because many subsections share a name
    ("Cleaning", "After a cleaning") and a bare FAQ answer means nothing without its question.
    Tables stay whole inside their piece. HTML comments, such as the file header, are removed.

    The database holds three tables: chunks (the pieces), chunks_fts (SQLite full-text search
    over heading and text, with Porter stemming, so "cancelling" finds "cancel") and chunks_vec
    (one embedding per piece, stored by sqlite-vec and compared by cosine distance, so only a
    vector's direction counts, not its length). The text embedded is the heading path plus
    the text, so a piece's heading counts towards its meaning too. With no embedder, the vector
    table is left out and only keyword search works.

    meta records a fingerprint of the documents and the splitting rules, and the embedding
    model's name. index_is_current compares both, so editing the handbook or changing the model
    always leads to a rebuild. Keyword search accepts an index with any model, since it never
    reads the vectors. Piece ids are unique across all documents, the first keeping the plain
    slug. A build writes to a temporary file and then replaces the index in
    one step, so a call never reads a half-built index.

Uses:
    sqlite-vec

Used by:
    factories.py, cli.py, clinic/retriever.py, evals/retrieval_eval.py
"""

import hashlib
import os
import re
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel

SEPARATOR = " › "  # Between the levels of a heading path.
SPLIT_VERSION = "2"  # Change it when the splitting rules change, so every index is rebuilt.
NO_MODEL = "none"  # The model name recorded for a keyword-only index.

HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


class Piece(BaseModel):
    """One searchable part of a document."""

    piece_id: str  # A stable slug of the heading path, such as finding-us/by-car.
    heading: str  # The heading path, such as "Finding us › By car".
    text: str
    source: str  # The document's file name.


class Embedder(Protocol):
    """What the index and the vector search need from an embedding model."""

    name: str
    dimensions: int

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per document text."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Return the vector of a search question."""
        ...


def split_sections(markdown: str, source: str, seen: set[str] | None = None) -> list[Piece]:
    """Return the pieces of one document, in order.

    seen holds the ids already used. build_index shares one set across every document, so two
    documents with the same heading never get the same id.
    """
    pieces: list[Piece] = []
    seen = set() if seen is None else seen
    path: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        buffer.clear()
        if text and path:
            pieces.append(_piece(path, text, source, seen))

    for line in COMMENT.sub("", markdown).splitlines():
        match = HEADING.match(line)
        if not match:
            buffer.append(line)
            continue
        flush()
        level, title = len(match.group(1)), match.group(2)
        # The document title stands alone. A ## heading starts a new path, and ### extends it.
        path = [title] if level < 3 else [*path[:1], title]
    flush()
    return pieces


def _piece(path: list[str], text: str, source: str, seen: set[str]) -> Piece:
    """Build one piece, giving it an id no piece in seen has."""
    base = "/".join(_slug(part) for part in path)
    piece_id, number = base, 2
    while piece_id in seen:
        piece_id, number = f"{base}-{number}", number + 1
    seen.add(piece_id)
    return Piece(piece_id=piece_id, heading=SEPARATOR.join(path), text=text, source=source)


def _slug(text: str) -> str:
    """Return lowercase words joined by hyphens, such as by-car."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def connect(path: Path) -> sqlite3.Connection:
    """Open an index with sqlite-vec loaded, so its vector table can be read and written."""
    import sqlite_vec

    connection = sqlite3.connect(path)
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)
    return connection


def fingerprint(docs: Iterable[Path]) -> str:
    """Return a hash of the splitting rules and every document's name and content."""
    digest = hashlib.sha256(SPLIT_VERSION.encode())
    for path in sorted(docs):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def index_is_current(rag_path: Path, docs: list[Path], model_name: str) -> bool:
    """Return True if the index exists and was built from these documents with this model.

    Keyword search (model_name NO_MODEL) works on any index, so it accepts one with vectors too,
    rather than rebuilding it without them under a call that still uses them.
    """
    if not rag_path.is_file():
        return False
    try:
        # closing, because a connection's own with block commits but never closes it.
        with closing(connect(rag_path)) as connection:
            meta = dict(connection.execute("SELECT key, value FROM meta").fetchall())
    except sqlite3.Error:
        return False  # A damaged or old-format index is rebuilt.
    same_model = model_name == NO_MODEL or meta.get("embedding_model") == model_name
    return meta.get("fingerprint") == fingerprint(docs) and same_model


def ensure_index(docs: list[Path], rag_path: Path, embedder: Embedder | None) -> bool:
    """Build the index if it is missing or out of date. Return True if it was built."""
    model_name = embedder.name if embedder else NO_MODEL
    if index_is_current(rag_path, docs, model_name):
        return False
    build_index(docs, rag_path, embedder)
    return True


def build_index(docs: list[Path], rag_path: Path, embedder: Embedder | None) -> int:
    """Write the whole index from these documents, and return how many pieces it holds."""
    seen: set[str] = set()  # Shared, so ids stay unique across documents.
    pieces = [piece for path in docs for piece in split_sections(path.read_text(), path.name, seen)]
    rag_path.parent.mkdir(parents=True, exist_ok=True)
    # A unique name, so builds in parallel runs of one process never write the same file.
    temporary = rag_path.with_name(f".{rag_path.name}.{uuid4().hex}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with closing(connect(temporary)) as connection, connection:
            _write(connection, docs, pieces, embedder)  # The inner with commits.
        os.replace(temporary, rag_path)  # One step, so a call never sees half an index.
    finally:
        temporary.unlink(missing_ok=True)
    return len(pieces)


def _write(
    connection: sqlite3.Connection,
    docs: list[Path],
    pieces: list[Piece],
    embedder: Embedder | None,
) -> None:
    """Create the tables and fill them."""
    connection.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE chunks (
          id INTEGER PRIMARY KEY,
          piece_id TEXT NOT NULL UNIQUE,
          source TEXT NOT NULL,
          heading TEXT NOT NULL,
          text TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
          heading, text, content='chunks', content_rowid='id',
          tokenize='porter unicode61 remove_diacritics 2'
        );
        """
    )
    connection.executemany(
        "INSERT INTO chunks (id, piece_id, source, heading, text) VALUES (?, ?, ?, ?, ?)",
        [
            (number, item.piece_id, item.source, item.heading, item.text)
            for number, item in enumerate(pieces, start=1)
        ],
    )
    connection.execute(
        "INSERT INTO chunks_fts (rowid, heading, text) SELECT id, heading, text FROM chunks"
    )
    meta = {
        "fingerprint": fingerprint(docs),
        "embedding_model": embedder.name if embedder else NO_MODEL,
        "pieces": str(len(pieces)),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if embedder is not None and pieces:
        import sqlite_vec

        vectors = embedder.embed_passages([embedding_text(item) for item in pieces])
        connection.execute(
            # Cosine distance compares direction only, so a vector's length never decides.
            "CREATE VIRTUAL TABLE chunks_vec USING vec0("
            f"embedding float[{embedder.dimensions}] distance_metric=cosine)"
        )
        connection.executemany(
            "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
            [
                (number, sqlite_vec.serialize_float32(vector))
                for number, vector in enumerate(vectors, start=1)
            ],
        )
        meta["dimensions"] = str(embedder.dimensions)
    connection.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta.items())


def embedding_text(piece: Piece) -> str:
    """Return the text embedded for a piece: its heading path, then its text."""
    return f"{piece.heading}\n{piece.text}"
