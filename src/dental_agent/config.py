"""Application settings. Stage: 1.

What this file is:
    Every value you can change without editing code: providers, models, timeouts, limits and
    file paths.

What it contains:
    - Settings, a pydantic-settings class that loads .env.
    - Provider choices: llm_provider, llm_model, embedding_provider, speech_backend,
      checkpointer.
    - Limits: turn_deadline_s (12), max_tool_rounds (3), max_turns (30), max_reply_sentences
      (3).
    - Voice tuning: eos_wait_ms, eos_wait_yes_ms, silence_timeout_s, yes_confidence_min.
    - The checker: checker_enabled (the on/off switch), checker_backends, the two model names,
      the two time limits and the cut-off.
      See docs/designs/checker_model.md.
    - Paths: calendar_path, rag_path, clinic_path, checkpoint_path, docs_dir, models_dir,
      runs_dir.
    - Keys as SecretStr fields, so they do not print by accident.

How to understand it:
    This is the only file that reads the environment. Other code receives a value from it, so a
    setting can be traced by searching this one class. To add a setting, add one field here and
    one line in .env.example.

Used by:
    factories.py, cli.py, and everything else through GraphDeps
"""

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """All settings. Each field reads the environment variable of the same name, in any case."""

    # Reads .env if it exists. Unknown names are ignored, so a shared .env causes no error.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Providers and models. factories.py turns these names into real objects.
    llm_provider: Literal["anthropic", "openrouter"] = "anthropic"
    llm_model: str = "claude-haiku-5-5"
    llm_timeout_s: float = 5  # How long one model call may take.
    # The ceiling for thinking plus the reply. The prompt keeps the reply short, and
    # reply_check trims it. Too low a ceiling cuts the model off while it is still thinking.
    llm_max_tokens: int = 2000
    # How hard an Anthropic model thinks. Low keeps a spoken turn fast.
    llm_effort: Literal["low", "medium", "high"] = "low"
    # Vector search embeds with FastEmbed, which runs small models on ONNX, with no PyTorch.
    # "none" means keyword search only, which the tests use so they never download a model.
    embedding_provider: Literal["fastembed", "none"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    speech_backend: Literal["deepgram", "local"] = "deepgram"
    checkpointer: Literal["memory", "sqlite"] = "memory"

    # The checker's on/off switch. Off means the guards decide with their word lists alone, as
    # before the checker existed. `dental-agent chat --checker on|off` overrides it for one call.
    checker_enabled: bool = True
    # The checker answers yes-or-no questions code decides. Backends are tried in this order, and
    # one without its key is left out. CHECKER_BACKENDS is comma-separated, and empty turns the
    # checker off. NoDecode stops pydantic-settings from expecting JSON for this list.
    checker_backends: Annotated[list[Literal["jev", "anthropic"]], NoDecode] = ["jev", "anthropic"]
    checker_jev_model: str = "typesafe/jev-1.14"  # A pinned version, not the moving alias.
    checker_anthropic_model: str = "claude-haiku-5-5"
    checker_jev_timeout_s: float = 1.0  # Measured p90 0.31 s, maximum 0.95 s.
    checker_anthropic_timeout_s: float = 2.0  # Measured p90 0.87 s.
    checker_cutoff: float = 0.5  # A Jev probability at or above this counts as yes. Not tuned.
    # Grey zones: a Jev score from the first number up to the second asks the next backend for
    # a second opinion. A reply judging the caller's own symptom scored 0.41 for advice on a
    # live call and passed; correct notice-period refusals scored 0.51 to 0.59 for a claim and
    # were replaced.
    checker_advice_grey_zone: tuple[float, float] = (0.3, 0.5)
    checker_claim_grey_zone: tuple[float, float] = (0.5, 0.65)

    # Keys. SecretStr hides the value if a setting is ever printed.
    anthropic_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    deepgram_api_key: SecretStr | None = None

    # Limits for a turn and a call.
    turn_deadline_s: float = 12  # One deadline for the whole turn.
    max_tool_rounds: int = 3  # Tool rounds allowed in one turn.
    max_turns: int = 30  # Caller turns allowed in one call.
    summary_deadline_s: float = 3  # How long the optional summary sentence may take.
    max_reply_sentences: int = 3  # A backstop. The prompt asks for 2.

    # Voice tuning, reviewed in Stage 2.
    eos_wait_ms: int = 700  # Silence that ends a normal reply.
    eos_wait_yes_ms: int = 1300  # Longer, so "yes... but Thursday" is not cut in half.
    silence_timeout_s: float = 7  # No speech for this long counts as silence.
    min_speech_ms: int = 200  # Anything shorter counts as no speech.
    yes_confidence_min: float = (
        0.7  # A less certain reply to a confirmation question cannot commit.
    )
    mic_reopen_ms: int = 300  # Pause after playback, so the bot does not hear itself.
    speech_timeout_s: float = 10  # transcribe and speak give up after this long.

    # File locations, relative to where the program runs.
    clinic_path: Path = Path("data/clinic.yaml")
    calendar_path: Path = Path("data/calendar.sqlite")
    rag_path: Path = Path("data/rag.sqlite")
    checkpoint_path: Path = Path("data/checkpoints.sqlite")
    docs_dir: Path = Path("data/docs")
    models_dir: Path = Path("data/models")  # Downloaded embedding models. Git ignores it.
    runs_dir: Path = Path("runs")

    @field_validator("checker_backends", mode="before")
    @classmethod
    def _split_backends(cls, value: Any) -> Any:
        """Read a comma-separated string, trimming spaces and dropping empty entries.

        A repeated name is kept once, in its first place, so the chain and the trace never list
        a backend twice.
        """
        if isinstance(value, str):
            value = [name.strip() for name in value.split(",") if name.strip()]
        if isinstance(value, list):
            return list(dict.fromkeys(value))
        return value  # Anything else is left for the type check to reject.
