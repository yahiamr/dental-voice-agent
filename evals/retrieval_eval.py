"""Scores document search on the labelled questions. Stage: 5.

What this file is:
    Runs every question in evals/retrieval_questions.yaml through keyword, vector and hybrid
    search over the patient handbook, and reports how often an answering piece comes back.
    Run it with uv run python -m evals.retrieval_eval, and add --save evals/results/retrieval.md
    to keep the result in Git. It needs no API key, only the embedding model.

What it contains:
    - load_questions(path), KINDS.
    - hit_at(ranking, expected, k) and reciprocal_rank(ranking, expected).
    - score(questions, rankings): hits, the scored count, mean reciprocal rank, hits by kind,
      and the ids of the misses.
    - main(): builds a fresh index in runs/retrieval/, scores each search, prints a page and,
      with --save, writes it.

How to understand it:
    The headline number is the hit rate at 3: the share of questions where at least one
    answering piece is among the 3 the model would see. Mean reciprocal rank, over the top 5,
    says how high the first answering piece sits. Unanswerable questions are listed but not
    scored, because every search returns its closest pieces anyway; what matters for those is
    whether the bot admits it does not know, which the scenarios check.

    The index is built here from data/docs with the configured embedding model, in a folder Git
    ignores, so the score never depends on, or changes, the index a call uses.

Uses:
    clinic/index.py, clinic/retriever.py, factories.py

Used by:
    You, tests/test_retrieval_eval.py
"""

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from dental_agent.clinic.index import build_index
from dental_agent.clinic.retriever import (
    RESULTS,
    HybridRetriever,
    KeywordRetriever,
    VectorRetriever,
)
from dental_agent.config import Settings
from dental_agent.factories import create_embeddings

QUESTIONS_PATH = Path(__file__).with_name("retrieval_questions.yaml")
INDEX_PATH = Path("runs/retrieval/rag.sqlite")
KINDS = ("direct", "reworded", "lookalike", "odd_place", "unanswerable")
DEPTH = 5  # How deep reciprocal rank looks.


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    """Read the labelled questions."""
    return yaml.safe_load(path.read_text())["questions"]


def hit_at(ranking: list[str], expected: set[str], k: int) -> bool:
    """Return True if any answering piece is in the top k."""
    return bool(expected & set(ranking[:k]))


def reciprocal_rank(ranking: list[str], expected: set[str]) -> float:
    """Return 1 over the position of the first answering piece, or 0 if none is there."""
    for position, piece_id in enumerate(ranking, start=1):
        if piece_id in expected:
            return 1.0 / position
    return 0.0


def score(questions: list[dict], rankings: dict[str, list[str]]) -> dict:
    """Score one search's rankings, leaving unanswerable questions out."""
    scored = [item for item in questions if item["expected"]]
    by_kind = {kind: {"hits": 0, "scored": 0} for kind in KINDS if kind != "unanswerable"}
    hits, ranks, misses = 0, [], []
    for item in scored:
        expected = set(item["expected"])
        ranking = rankings[item["id"]]
        hit = hit_at(ranking, expected, RESULTS)
        hits += hit
        ranks.append(reciprocal_rank(ranking[:DEPTH], expected))
        by_kind[item["kind"]]["scored"] += 1
        by_kind[item["kind"]]["hits"] += hit
        if not hit:
            misses.append(item["id"])
    return {
        "hits": hits,
        "scored": len(scored),
        "mrr": sum(ranks) / len(ranks) if ranks else 0.0,
        "by_kind": by_kind,
        "misses": misses,
    }


def _rankings(retriever, questions: list[dict]) -> dict[str, list[str]]:
    """Return each question's top piece ids from one search."""
    return {
        item["id"]: [result.piece_id for result in retriever.rank(item["question"], DEPTH)]
        for item in questions
    }


def _page(questions: list[dict], results: dict[str, dict], tops: dict, model: str) -> str:
    """Write the results page."""
    scored = next(iter(results.values()))["scored"]
    lines = [
        "# Retrieval results",
        "",
        (
            f"Document: `data/docs/patient-handbook.md`. Embedding model: `{model}`. "
            f"Commit: `{_git_commit()}`. Run on {datetime.now():%Y-%m-%d %H:%M}."
        ),
        "",
        (
            f"{scored} labelled questions in the caller's own words, plus "
            f"{len(questions) - scored} the handbook does not answer. A hit means an answering "
            f"piece is among the {RESULTS} the model would see."
        ),
        "",
        "| Search | Hits at 3 | Hit rate | MRR at 5 | " + " | ".join(KINDS[:-1]) + " |",
        "| --- | --- | --- | --- | " + " | ".join("---" for _ in KINDS[:-1]) + " |",
    ]
    for name, result in results.items():
        kinds = " | ".join(
            f"{result['by_kind'][kind]['hits']} of {result['by_kind'][kind]['scored']}"
            for kind in KINDS[:-1]
        )
        lines.append(
            f"| {name} | {result['hits']} of {result['scored']} | "
            f"{result['hits'] / result['scored']:.0%} | {result['mrr']:.2f} | {kinds} |"
        )
    lines += ["", "**Misses** (no answering piece in the top 3):", ""]
    text = {item["id"]: item["question"] for item in questions}
    for name, result in results.items():
        missed = ", ".join(f'{qid} "{text[qid]}"' for qid in result["misses"]) or "none"
        lines.append(f"- {name}: {missed}")
    lines += ["", "**Unanswerable questions**, with hybrid search's top piece:", ""]
    for item in questions:
        if not item["expected"]:
            lines.append(f'- {item["id"]} "{item["question"]}": `{tops[item["id"]] or "nothing"}`')
    return "\n".join(lines) + "\n"


def _git_commit() -> str:
    """Return the current commit, marked -dirty when there are uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--exclude=*"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "none"
    return result.stdout.strip()


def main() -> None:
    """Build a fresh index, score the three searches, print the page and maybe save it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--save", type=Path, help="Also save the page, such as evals/results/retrieval.md."
    )
    args = parser.parse_args()

    settings = Settings().model_copy(update={"embedding_provider": "fastembed"})
    embedder = create_embeddings(settings)
    if embedder is None:
        raise SystemExit("The embedding model could not load, so vector search cannot be scored.")
    docs = sorted(Path(settings.docs_dir).glob("*.md"))
    build_index(docs, INDEX_PATH, embedder)

    questions = load_questions()
    searches = {
        "keyword": KeywordRetriever(INDEX_PATH),
        "vector": VectorRetriever(INDEX_PATH, embedder),
        "hybrid": HybridRetriever(INDEX_PATH, embedder),
    }
    rankings = {name: _rankings(search, questions) for name, search in searches.items()}
    results = {name: score(questions, ranking) for name, ranking in rankings.items()}
    tops = {qid: (ranking[0] if ranking else "") for qid, ranking in rankings["hybrid"].items()}
    page = _page(questions, results, tops, embedder.name)
    print(page)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(page)
        print(f"Saved to {args.save}.")


if __name__ == "__main__":
    main()
