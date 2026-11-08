"""Test doubles. Stage: 1.

What this file is:
    Stand-ins for the model, speech and calendar, so tests need no network or keys.

What it contains:
    - ScriptedModel: plays a list of model responses.
    - FakeSpeech: scripted input, and it records what was spoken.
    - FailingCalendar: raises before or after a write.
    - FakeEmbeddings: fixed vectors.
    - ScriptedChecker: a checker backend that plays a list of answers or errors.

How to understand it:
    ScriptedModel is the key one. It can act like a model that invents a slot ID, claims a false
    booking or skips the confirmation question, which proves the guards hold even when the model
    misbehaves.

Used by:
    pytest
"""

from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from dental_agent.booking.calendar import Calendar, CalendarError
from dental_agent.checker.base import CheckAnswer
from dental_agent.speech.base import SpeakResult, SpeechResult


def text_reply(text: str) -> AIMessage:
    """Build a normal model reply with no tool calls."""
    return AIMessage(content=text, response_metadata={"stop_reason": "end_turn"})


def tool_reply(*calls: tuple[str, dict[str, Any]], text: str = "") -> AIMessage:
    """Build a model response asking for tools. Each call is a (name, args) pair.

    text is what the model writes next to its tool calls, such as a lead-in sentence.
    """
    tool_calls = [
        {"id": f"call-{index}", "name": name, "args": args, "type": "tool_call"}
        for index, (name, args) in enumerate(calls)
    ]
    return AIMessage(
        content=text, tool_calls=tool_calls, response_metadata={"stop_reason": "tool_use"}
    )


def cut_off_reply(text: str = "I was going to") -> AIMessage:
    """Build a response that stopped because it ran out of tokens."""
    return AIMessage(content=text, response_metadata={"stop_reason": "max_tokens"})


class ScriptedModel(BaseChatModel):
    """A chat model that plays a list of responses.

    It stands in for a real model, including one that misbehaves, so tests prove the guards hold
    with no API key. Running out of responses raises an error, which also stands in for a failed
    model call.
    """

    script: list[AIMessage]
    # Every message list the model received, so tests can inspect what it was shown.
    seen: list[list[Any]] = Field(default_factory=list)
    # Runs during each model call, so a test can move the clock past a deadline.
    on_call: Callable[[], None] | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        """Return itself, because a script already decides which tools are called."""
        return self

    def _generate(
        self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        """Return the next scripted response."""
        self.seen.append(messages)
        if self.on_call:
            self.on_call()
        if not self.script:
            raise RuntimeError("ScriptedModel ran out of responses")
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


class FailingCalendar(Calendar):
    """A real calendar whose book method fails on request.

    "before" fails with nothing written. "after" fails once the row is safely stored, like a
    connection lost after the database committed.
    """

    fail_book: str | None = None
    fail_reads: bool = False

    def book(self, *args: Any, **kwargs: Any) -> Any:
        """Write a booking, or fail as the test asked."""
        if self.fail_book == "before":
            raise CalendarError("write failed")
        record = super().book(*args, **kwargs)
        if self.fail_book == "after":
            raise CalendarError("connection lost after the write")
        return record

    def actions_for_call(self, conversation_id: str) -> Any:
        """Read actions, or fail as the test asked."""
        if self.fail_reads:
            raise CalendarError("read failed")
        return super().actions_for_call(conversation_id)


class FakeSpeech:
    """A Speech object that plays scripted input and records what was spoken."""

    uses_tone = False

    def __init__(self, inputs: list[Any]) -> None:
        # Each input is caller text, or a ready-made SpeechResult for silence and errors.
        self.inputs = list(inputs)
        self.spoken: list[str] = []
        self.confirmations: list[str] = []
        self.expect_yes_flags: list[bool] = []
        self.fail_speak = False  # When true, every speak call fails.
        self.fail_confirmations = False  # When true, only confirmation questions fail.
        self.fail_after: int | None = None  # When set, speak fails after this many lines.
        self.raise_on_transcribe: BaseException | None = None

    def transcribe(self, expect_yes: bool) -> SpeechResult:
        """Return the next input. Running out counts as the caller hanging up."""
        self.expect_yes_flags.append(expect_yes)
        if not self.inputs:
            return SpeechResult(kind="closed")
        item = self.inputs.pop(0)
        # A function in the list runs between turns, so a test can change the world mid-call.
        if callable(item):
            item()
            return self.transcribe(expect_yes)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, SpeechResult):
            return item
        return SpeechResult(kind="speech", text=item, confidence=1.0)

    def speak(self, text: str, asks_confirmation: bool = False) -> SpeakResult:
        """Record the line, or fail as the test asked."""
        failing = (
            self.fail_speak
            or (self.fail_confirmations and asks_confirmation)
            or (self.fail_after is not None and len(self.spoken) >= self.fail_after)
        )
        if failing:
            return SpeakResult(ok=False)
        self.spoken.append(text)
        if asks_confirmation:
            self.confirmations.append(text)
        return SpeakResult(ok=True)


class ScriptExhausted(BaseException):
    """A scripted fake ran out of results.

    It is a BaseException, not an Exception, so the checker chain, which catches every
    Exception, can't turn an unexpected extra check into a quiet strike.
    """


class ScriptedChecker:
    """A checker backend that plays a list of answers or errors, in order.

    Tests wrap it in a real CheckerChain, so the chain's skip and fallback logic is tested too.
    Running out of results raises ScriptExhausted, which the chain doesn't catch, so an
    unexpected extra check fails the test. calls, limits, texts, roles and questions record what
    was asked.
    """

    def __init__(
        self,
        name: str,
        results: list[CheckAnswer | Exception],
        on_call: Callable[[], None] | None = None,
    ) -> None:
        self.name = name
        self.results = list(results)
        self.on_call = on_call  # Runs during each call, so a test can move a clock.
        self.calls = 0
        self.limits: list[float] = []
        self.texts: list[str] = []
        self.roles: list[str] = []
        self.questions: list[dict[str, str]] = []

    def ask(
        self, text: str, questions: dict[str, str], role: str, time_limit_s: float
    ) -> CheckAnswer:
        """Return or raise the next scripted result."""
        self.calls += 1
        self.limits.append(time_limit_s)
        self.texts.append(text)
        self.roles.append(role)
        self.questions.append(questions)
        if self.on_call:
            self.on_call()
        if not self.results:
            raise ScriptExhausted(f"ScriptedChecker {self.name} ran out of results")
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result
