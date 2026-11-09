"""Tests for the factories. Stage: 1.

What this file is:
    Proves provider setup, with no network calls.

What it contains:
    - create_llm returns a chat model with tool binding.
    - create_checkpointer defaults to in-memory.
    - create_embeddings builds FastEmbed, or returns None when it is missing or cannot load.
    - create_retriever builds the index and picks hybrid, keyword or no search.
    - normalise_stop treats unknown values as cut_off.
    - A missing key or extra exits naming it.
    - create_checker builds only the backends whose key is present, and CHECKER_BACKENDS is read
      as a comma-separated list.

Used by:
    pytest
"""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError
from tests.fakes import FakeEmbedder

from dental_agent.clinic.retriever import HybridRetriever, KeywordRetriever, NoRetriever
from dental_agent.config import Settings
from dental_agent.deps import normalise_stop
from dental_agent.factories import (
    build_deps,
    create_checker,
    create_checkpointer,
    create_embeddings,
    create_llm,
    create_retriever,
    create_speech,
)
from dental_agent.speech.deepgram import DeepgramSpeech
from dental_agent.speech.display import CallDisplay
from dental_agent.speech.text import TextSpeech


def settings(**changes):
    """Build settings that ignore any .env or environment variables on this machine."""
    return Settings(_env_file=None, **changes)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Remove real keys, so these tests behave the same on every machine."""
    for name in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "DEEPGRAM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    # A developer's own checker settings would change the checker tests below.
    for name in (
        "CHECKER_ENABLED",
        "CHECKER_BACKENDS",
        "CHECKER_JEV_MODEL",
        "CHECKER_ANTHROPIC_MODEL",
        "CHECKER_JEV_TIMEOUT_S",
        "CHECKER_ANTHROPIC_TIMEOUT_S",
        "CHECKER_CUTOFF",
    ):
        monkeypatch.delenv(name, raising=False)


def test_create_llm_returns_a_chat_model_that_can_bind_tools():
    # This needs no network, because nothing is sent until the model is called.
    model = create_llm(settings(anthropic_api_key="test-key"))
    assert model.bind_tools([]) is not None


def test_create_llm_sets_a_timeout_and_one_retry():
    model = create_llm(settings(anthropic_api_key="test-key", llm_timeout_s=4))
    assert model.default_request_timeout == 4
    assert model.max_retries == 1


def test_create_llm_asks_for_low_effort():
    # Voice replies need speed, and thinking at the default effort used up all of max_tokens.
    model = create_llm(settings(anthropic_api_key="test-key"))
    payload = model._get_request_payload("hi")
    assert payload["output_config"]["effort"] == "low"


def test_max_tokens_leaves_room_for_thinking():
    # max_tokens covers thinking as well as the reply. 300 was cut off on a real call.
    assert settings().llm_max_tokens >= 2000


def test_a_missing_key_exits_naming_it():
    with pytest.raises(SystemExit) as caught:
        create_llm(settings())
    assert "ANTHROPIC_API_KEY" in str(caught.value)


def test_a_missing_openrouter_key_exits_naming_it():
    with pytest.raises(SystemExit) as caught:
        create_llm(settings(llm_provider="openrouter"))
    assert "OPENROUTER_API_KEY" in str(caught.value)


def test_a_missing_openrouter_extra_exits_with_the_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "langchain_openrouter", None)
    with pytest.raises(SystemExit) as caught:
        create_llm(settings(llm_provider="openrouter", openrouter_api_key="test-key"))
    assert "uv sync --extra openrouter" in str(caught.value)


def test_create_checkpointer_defaults_to_memory():
    assert isinstance(create_checkpointer(settings()), InMemorySaver)


def test_a_missing_sqlite_extra_exits_with_the_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.sqlite", None)
    with pytest.raises(SystemExit) as caught:
        create_checkpointer(settings(checkpointer="sqlite"))
    assert "uv sync --extra checkpoint-sqlite" in str(caught.value)


def fake_fastembed(monkeypatch, fail=False):
    """Install a stand-in fastembed module, so no test downloads a model."""
    module = ModuleType("fastembed")

    class TextEmbedding:
        @staticmethod
        def list_supported_models():
            return [{"model": "BAAI/bge-small-en-v1.5", "dim": 384}]

        def __init__(self, model_name, cache_dir=None):
            if fail:
                raise OSError("no network")
            self.model_name, self.cache_dir = model_name, cache_dir

        def passage_embed(self, texts):
            return ([1.0] * 384 for _ in texts)

        def query_embed(self, text):
            return iter([[0.5] * 384])

    module.TextEmbedding = TextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", module)


def test_create_embeddings_builds_fastembed_with_the_configured_model(monkeypatch, tmp_path):
    fake_fastembed(monkeypatch)
    embedder = create_embeddings(
        settings(embedding_provider="fastembed", models_dir=tmp_path / "models")
    )
    assert embedder.name == "BAAI/bge-small-en-v1.5"
    assert embedder.dimensions == 384
    assert len(embedder.embed_passages(["a", "b"])) == 2
    assert len(embedder.embed_query("a")) == 384


def test_create_embeddings_returns_none_when_fastembed_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)
    assert create_embeddings(settings(embedding_provider="fastembed")) is None


def test_create_embeddings_returns_none_when_the_model_cannot_load(monkeypatch, tmp_path):
    # With no network on the first run, search falls back to keywords rather than failing.
    fake_fastembed(monkeypatch, fail=True)
    no_network = settings(embedding_provider="fastembed", models_dir=tmp_path / "models")
    assert create_embeddings(no_network) is None


def test_the_none_provider_means_keyword_search_only():
    assert create_embeddings(settings(embedding_provider="none")) is None


def search_settings(tmp_path, text="# H\n\n## By car\n\nThere is no parking.\n"):
    """Settings whose documents and index live in tmp_path."""
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "handbook.md").write_text(text)
    return settings(docs_dir=docs, rag_path=tmp_path / "rag.sqlite")


def test_create_retriever_builds_keyword_search_with_no_embeddings(tmp_path):
    retriever = create_retriever(search_settings(tmp_path), None)
    assert isinstance(retriever, KeywordRetriever)
    assert retriever.search("parking")[0].heading == "By car"


def test_create_retriever_builds_hybrid_search_with_embeddings(tmp_path):
    retriever = create_retriever(search_settings(tmp_path), FakeEmbedder())
    assert isinstance(retriever, HybridRetriever)
    assert retriever.search("car")[0].heading == "By car"


def test_create_retriever_has_no_search_without_documents(tmp_path):
    empty = settings(docs_dir=tmp_path / "none", rag_path=tmp_path / "rag.sqlite")
    assert isinstance(create_retriever(empty, None), NoRetriever)


def test_create_retriever_falls_back_to_keywords_when_embedding_fails(tmp_path):
    class BrokenEmbedder(FakeEmbedder):
        def embed_passages(self, texts):
            raise RuntimeError("model crashed")

    retriever = create_retriever(search_settings(tmp_path), BrokenEmbedder())
    assert isinstance(retriever, KeywordRetriever)
    assert retriever.search("parking")


def test_text_mode_always_gets_text_speech():
    assert isinstance(create_speech(settings(), "text"), TextSpeech)


@pytest.fixture
def voice_extra(monkeypatch):
    """Stand in for the voice extra, so these tests pass without deepgram-sdk or sounddevice.

    `uv sync` leaves the extra out, so a reviewer's first test run must not need it.
    """
    deepgram = ModuleType("deepgram")
    deepgram.DeepgramClient = lambda **options: SimpleNamespace(options=options)
    monkeypatch.setitem(sys.modules, "deepgram", deepgram)
    monkeypatch.setitem(sys.modules, "sounddevice", ModuleType("sounddevice"))


def test_voice_mode_gets_deepgram_with_the_key_terms_and_fixture_folder(tmp_path, voice_extra):
    speech = create_speech(
        settings(deepgram_api_key="test-key"), "voice", ["Mirador"], tmp_path / "fixtures"
    )
    assert isinstance(speech, DeepgramSpeech)
    assert speech.keyterms == ["Mirador"]
    assert speech.fixture_dir == tmp_path / "fixtures"


def test_a_voice_call_shows_itself_in_the_terminal(voice_extra):
    speech = create_speech(settings(deepgram_api_key="test-key"), "voice")
    assert isinstance(speech.display, CallDisplay)


def test_voice_mode_exits_naming_the_missing_deepgram_key(voice_extra):
    with pytest.raises(SystemExit) as caught:
        create_speech(settings(), "voice")
    assert "DEEPGRAM_API_KEY" in str(caught.value)


def test_voice_mode_exits_with_the_install_command_without_the_extra(monkeypatch):
    # A None entry in sys.modules makes the import fail, as it would without the extra.
    monkeypatch.setitem(sys.modules, "deepgram", None)
    with pytest.raises(SystemExit) as caught:
        create_speech(settings(deepgram_api_key="test-key"), "voice")
    assert "uv sync --extra voice-deepgram" in str(caught.value)


def test_local_speech_exits_because_it_is_not_built():
    with pytest.raises(SystemExit) as caught:
        create_speech(settings(speech_backend="local", deepgram_api_key="test-key"), "voice")
    assert "SPEECH_BACKEND=deepgram" in str(caught.value)


def test_build_deps_in_voice_mode_boosts_the_clinic_and_service_names(tmp_path, voice_extra):
    deps = build_deps(
        settings(
            anthropic_api_key="test-key",
            deepgram_api_key="test-key",
            calendar_path=tmp_path / "calendar.sqlite",
        ),
        "voice",
        fixture_dir=tmp_path / "fixtures",
    )
    assert isinstance(deps.speech, DeepgramSpeech)
    assert deps.speech.keyterms[0] == "Mirador Dental"
    assert "check-up" in deps.speech.keyterms
    assert deps.speech.fixture_dir == tmp_path / "fixtures"


def test_build_deps_holds_every_service(tmp_path):
    deps = build_deps(
        settings(
            anthropic_api_key="test-key",
            calendar_path=tmp_path / "calendar.sqlite",
        ),
        "text",
    )
    assert isinstance(deps.speech, TextSpeech)
    assert deps.graph.clinic.name == "Mirador Dental"
    assert isinstance(deps.graph.checkpointer, InMemorySaver)


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"stop_reason": "end_turn"}, "complete"),  # Anthropic.
        ({"stop_reason": "tool_use"}, "complete"),
        ({"stop_reason": "max_tokens"}, "cut_off"),
        ({"finish_reason": "stop"}, "complete"),  # OpenRouter.
        ({"finish_reason": "tool_calls"}, "complete"),
        ({"finish_reason": "length"}, "cut_off"),
        ({"stop_reason": "something_new"}, "cut_off"),
        ({}, "cut_off"),
    ],
)
def test_normalise_stop(metadata, expected):
    assert normalise_stop(AIMessage(content="", response_metadata=metadata)) == expected


# ---------- the checker ----------


def test_create_checker_builds_both_backends_in_order_when_both_keys_are_set():
    chain = create_checker(settings(openrouter_api_key="or-key", anthropic_api_key="a-key"))
    assert chain.names == ["jev", "anthropic"]
    assert chain.time_limits == {"jev": 1.0, "anthropic": 2.0}


def test_create_checker_leaves_out_a_backend_with_no_key():
    assert create_checker(settings(anthropic_api_key="a-key")).names == ["anthropic"]
    assert create_checker(settings(openrouter_api_key="or-key")).names == ["jev"]


def test_create_checker_treats_an_empty_key_as_missing():
    # .env.example ships empty keys, which load as an empty string, not as nothing.
    chain = create_checker(settings(openrouter_api_key="", anthropic_api_key="a-key"))
    assert chain.names == ["anthropic"]


def test_an_empty_checker_backends_setting_gives_an_empty_chain(monkeypatch):
    monkeypatch.setenv("CHECKER_BACKENDS", "")
    chain = create_checker(settings(openrouter_api_key="or-key", anthropic_api_key="a-key"))
    assert chain.names == []


def test_checker_backends_is_read_as_a_comma_separated_list(monkeypatch):
    monkeypatch.setenv("CHECKER_BACKENDS", " anthropic , jev ")
    assert settings().checker_backends == ["anthropic", "jev"]
    assert settings(checker_backends=["jev"]).checker_backends == ["jev"]


def test_a_repeated_checker_backend_is_kept_once_in_its_first_place(monkeypatch):
    monkeypatch.setenv("CHECKER_BACKENDS", "jev,anthropic,jev")
    assert settings().checker_backends == ["jev", "anthropic"]
    assert settings(checker_backends=["anthropic", "anthropic"]).checker_backends == ["anthropic"]


def test_an_unknown_checker_backend_is_a_settings_error(monkeypatch):
    monkeypatch.setenv("CHECKER_BACKENDS", "jev,gpt")
    with pytest.raises(ValidationError):
        settings()


def test_the_checker_switch_turns_the_checker_off_even_with_keys():
    off = settings(checker_enabled=False, openrouter_api_key="or-key", anthropic_api_key="a-key")
    assert create_checker(off).names == []


def test_checker_enabled_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("CHECKER_ENABLED", "false")
    assert settings().checker_enabled is False
    monkeypatch.setenv("CHECKER_ENABLED", "true")
    assert settings().checker_enabled is True


def test_tests_never_use_the_real_search_index():
    # Running the suite once wrote a keyword-only data/rag.sqlite into the project.
    real = (Path(__file__).parent.parent / "data").resolve()
    assert real not in Settings().rag_path.resolve().parents
    assert real not in Settings().models_dir.resolve().parents


def test_the_chain_asks_for_a_second_opinion_near_the_cut_off():
    chain = create_checker(settings(openrouter_api_key="or-key", anthropic_api_key="an-key"))
    assert chain.second_opinion == {"medical_advice": (0.3, 0.5), "claims_change": (0.5, 0.65)}
