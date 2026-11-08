"""The embedding model for vector search. Stage: 5.

What this file is:
    A small wrapper around FastEmbed, giving the index and the vector search one interface.

What it contains:
    - FastEmbedder(model_name, cache_dir): name, dimensions, embed_passages(texts) and
      embed_query(text).

How to understand it:
    FastEmbed runs small embedding models on ONNX Runtime, with no PyTorch, so it installs
    lightly and runs fast on a laptop CPU. The default model, BAAI/bge-small-en-v1.5, turns a
    text into 384 numbers. The first use downloads it (about 64 MB) into data/models/, which
    Git ignores. After that it loads in about a tenth of a second, embeds the handbook in a
    fraction of a second and a question in a few milliseconds.

    Documents and questions are embedded differently on purpose: BGE models expect a short
    instruction before a search question, and FastEmbed's query_embed adds it, while
    passage_embed does not. Using the right one for each side gives better matches.

    fastembed is imported when a FastEmbedder is made, so the rest of the package loads without
    it. factories.create_embeddings catches any failure and falls back to keyword search.

Uses:
    fastembed

Used by:
    factories.py
"""

from pathlib import Path


class FastEmbedder:
    """An embedding model run by FastEmbed."""

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        from fastembed import TextEmbedding

        sizes = {item["model"]: item["dim"] for item in TextEmbedding.list_supported_models()}
        if model_name not in sizes:
            raise ValueError(f"FastEmbed does not support the model {model_name}.")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(model_name, cache_dir=str(cache_dir))
        self.name = model_name
        self.dimensions = sizes[model_name]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per document text."""
        return [[float(value) for value in vector] for vector in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        """Return the vector of a search question, with the model's search instruction."""
        vector = next(iter(self._model.query_embed(text)))
        return [float(value) for value in vector]
