"""Builds the search index. Stage: 5.

What this file is:
    Turns the Markdown documents into a searchable database.

What it will contain:
    - build_index(docs_dir, rag_path, embeddings=None).

How to understand it:
    It rebuilds data/rag.sqlite from nothing on each run, so it is never partly out of date.
    Documents split at ## headings, and each section becomes one chunk. If embeddings are
    passed, it adds vectors and records the model name, so the search can detect a mismatch
    later.

Used by:
    cli.py
"""
