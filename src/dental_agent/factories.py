"""Builds every outside service. Stage: 1, 2 and 5.

What this file is:
    The only file that maps provider names to real implementations. The rest of the code
    receives ready objects.

What it contains:
    - create_llm: a chat model from init_chat_model, with a timeout and one SDK retry.
    - create_embeddings: an embedding model from init_embeddings, or nothing if not installed.
    - create_retriever: hybrid search, or keyword search with no embeddings.
    - create_checkpointer: in-memory by default, SQLite with an optional extra.
    - create_speech: text mode always gets TextSpeech, and voice mode gets DeepgramSpeech.
    - create_checker: the checker chain, with each listed backend whose key is set.
    - build_deps(settings, mode, fixture_dir): calls these and returns one Deps object.

How to understand it:
    To switch provider you change .env, not code. To add a provider you edit this file only. A
    missing key or extra exits here with a message naming it. Tests skip this file and build
    Deps from fakes, so test code stays out of production config.

Uses:
    config.py, deps.py, clinic/retriever.py, speech/

Used by:
    cli.py, evals/evaluate.py
"""

import sqlite3
from pathlib import Path
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from dental_agent.booking.calendar import Calendar
from dental_agent.checker import CheckerChain
from dental_agent.checker.anthropic import AnthropicChecker
from dental_agent.checker.jev import JevChecker
from dental_agent.clinic.data import load_clinic
from dental_agent.clinic.retriever import NoRetriever, Retriever
from dental_agent.clock import Clock
from dental_agent.config import Settings
from dental_agent.deps import Deps, GraphDeps
from dental_agent.models import (
    Appointment,
    Details,
    EndReason,
    FixedReason,
    Offer,
    PendingAction,
    ToolExit,
    UrgentLevel,
)
from dental_agent.speech.base import Speech
from dental_agent.speech.text import TextSpeech
from dental_agent.storage.trace import TraceSink

# One retry covers a brief network problem, without doubling a long wait many times.
LLM_RETRIES = 1

# The types CallState stores. LangGraph loads only listed types from a checkpoint, so a saved
# file can never create an object of a type we did not choose.
CHECKPOINT_TYPES = (
    Appointment,
    Details,
    EndReason,
    FixedReason,
    Offer,
    PendingAction,
    ToolExit,
    UrgentLevel,
)


def create_llm(settings: Settings) -> Any:
    """Build the chat model chosen in settings, or exit naming what is missing."""
    # Settings that only one provider understands.
    provider_options: dict[str, Any] = {}
    if settings.llm_provider == "openrouter":
        key = _require_key(settings.openrouter_api_key, "OPENROUTER_API_KEY")
        try:
            import langchain_openrouter  # noqa: F401
        except ImportError:
            raise SystemExit(
                "OpenRouter needs its extra. Run: uv sync --extra openrouter"
            ) from None
    else:
        key = _require_key(settings.anthropic_api_key, "ANTHROPIC_API_KEY")
        # Lower effort, not disabled thinking: with thinking off, a model can write a tool
        # call as plain text, and that call would never run.
        provider_options["output_config"] = {"effort": settings.llm_effort}

    return init_chat_model(
        settings.llm_model,
        model_provider=settings.llm_provider,
        api_key=key,
        timeout=settings.llm_timeout_s,
        max_retries=LLM_RETRIES,
        max_tokens=settings.llm_max_tokens,
        **provider_options,
    )


def create_embeddings(settings: Settings) -> Any | None:
    """Build the embedding model, or return None if the embeddings extra is not installed."""
    try:
        import langchain_huggingface  # noqa: F401
    except ImportError:
        return None
    # Imported here, because this path needs the extra.
    from langchain.embeddings import init_embeddings

    return init_embeddings(settings.embedding_model, provider=settings.embedding_provider)


def create_retriever(settings: Settings, embeddings: Any | None) -> Retriever:
    """Build document search. Stage 5 adds keyword and hybrid search here."""
    return NoRetriever()


def create_checkpointer(settings: Settings) -> Any:
    """Build the store LangGraph uses to keep call state between turns."""
    allowed = [(item.__module__, item.__name__) for item in CHECKPOINT_TYPES]
    serde = JsonPlusSerializer(allowed_msgpack_modules=allowed)
    if settings.checkpointer == "memory":
        return InMemorySaver(serde=serde)
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        raise SystemExit(
            "The SQLite checkpointer needs its extra. Run: uv sync --extra checkpoint-sqlite"
        ) from None
    # LangGraph may run nodes on other threads, so the connection must allow that.
    settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(settings.checkpoint_path, check_same_thread=False)
    return SqliteSaver(connection, serde=serde)


def create_speech(
    settings: Settings,
    mode: Literal["text", "voice"],
    keyterms: list[str] | None = None,
    fixture_dir: Path | str | None = None,
) -> Speech:
    """Build the speech backend. Text mode always gets TextSpeech.

    keyterms are words Deepgram should listen for, such as the clinic's name. fixture_dir, if
    set, is where each voice turn is saved.
    """
    if mode == "text":
        return TextSpeech()
    if settings.speech_backend == "local":
        raise SystemExit(
            "Local speech is an optional stage that is not built. Set SPEECH_BACKEND=deepgram."
        )
    try:
        import deepgram  # noqa: F401
        import sounddevice  # noqa: F401
    except (ImportError, OSError):  # sounddevice raises OSError if PortAudio is missing.
        raise SystemExit("Voice needs its extra. Run: uv sync --extra voice-deepgram") from None
    key = _require_key(settings.deepgram_api_key, "DEEPGRAM_API_KEY")
    # Imported here, so text mode never loads the voice backend.
    from dental_agent.speech.deepgram import DeepgramSpeech

    return DeepgramSpeech(settings, key, keyterms or [], fixture_dir)


def create_checker(settings: Settings) -> CheckerChain:
    """Build the checker chain from each listed backend whose key is set and not empty.

    A missing key leaves that backend out instead of stopping the program, because the checker
    is optional: with no backend at all, the fixed rules decide alone. checker_enabled false
    gives an empty chain, whatever keys are set.
    """
    if not settings.checker_enabled:
        return CheckerChain([], {})
    backends = []
    for name in settings.checker_backends:
        if name == "jev" and _has_key(settings.openrouter_api_key):
            key = settings.openrouter_api_key.get_secret_value()
            backends.append(JevChecker(settings.checker_jev_model, key, settings.checker_cutoff))
        elif name == "anthropic" and _has_key(settings.anthropic_api_key):
            key = settings.anthropic_api_key.get_secret_value()
            backends.append(AnthropicChecker(settings.checker_anthropic_model, key))
    limits = {
        "jev": settings.checker_jev_timeout_s,
        "anthropic": settings.checker_anthropic_timeout_s,
    }
    return CheckerChain(backends, {backend.name: limits[backend.name] for backend in backends})


def build_deps(
    settings: Settings, mode: Literal["text", "voice"], fixture_dir: Path | str | None = None
) -> Deps:
    """Build every service a call needs. fixture_dir saves each voice turn, if set."""
    # The clinic file is only read, so loading it first creates no data. Its names are the
    # words Deepgram should listen for.
    clinic = load_clinic(settings.clinic_path)
    keyterms = [clinic.name] + [service.name for service in clinic.services]
    # Speech and the model come first, so a missing key exits before any data is created.
    speech = create_speech(settings, mode, keyterms, fixture_dir)
    model = create_llm(settings)
    embeddings = create_embeddings(settings)
    graph = GraphDeps(
        model=model,
        checkpointer=create_checkpointer(settings),
        # The calendar has no factory, because SQLite is its only implementation.
        calendar=Calendar(settings.calendar_path),
        clock=Clock(),
        clinic=clinic,
        retriever=create_retriever(settings, embeddings),
        trace=TraceSink(),
        settings=settings,
        checker=create_checker(settings),
    )
    return Deps(graph=graph, speech=speech)


def _has_key(key: Any) -> bool:
    """Return True if a key is set and not empty, the same rule _require_key uses."""
    return key is not None and bool(key.get_secret_value())


def _require_key(key: Any, name: str) -> str:
    """Return a key's value, or exit naming the setting that is missing."""
    if key is None or not key.get_secret_value():
        raise SystemExit(f"{name} is missing. Add it to .env. See .env.example.")
    return key.get_secret_value()
