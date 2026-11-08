"""Containers for ready-made services. Stage: 1.

What this file is:
    Holds the objects the rest of the code needs, and turns provider stop reasons into one
    value.

What it contains:
    - GraphDeps: model, checkpointer, calendar, clock, clinic data, retriever, trace sink,
      settings, the checker chain, and the emergency check in flight.
    - Deps: GraphDeps plus speech.
    - normalise_stop(response): returns complete or cut_off.

How to understand it:
    Instead of each function importing a provider, it receives these objects. That is how a test
    swaps the real model for ScriptedModel. The graph gets GraphDeps with no speech, because the
    graph handles text only. normalise_stop treats any unknown stop reason as cut_off, so a
    doubtful response never runs tools.

Uses:
    models.py

Used by:
    factories.py, graph/, tools/, session.py
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage

from dental_agent.booking.calendar import Calendar
from dental_agent.checker import CheckerChain
from dental_agent.clinic.data import Clinic
from dental_agent.clinic.retriever import Retriever
from dental_agent.clock import Clock
from dental_agent.config import Settings
from dental_agent.storage.trace import TraceSink

# Stop reasons that mean the model finished what it wanted to say.
COMPLETE_STOPS = {"end_turn", "tool_use", "stop_sequence", "stop", "tool_calls"}


@dataclass
class GraphDeps:
    """Everything the graph needs. It holds no speech, because the graph handles text only."""

    model: Any  # A LangChain chat model.
    checkpointer: Any  # A LangGraph checkpointer, which keeps CallState between turns.
    calendar: Calendar
    clock: Clock
    clinic: Clinic
    retriever: Retriever
    trace: TraceSink
    settings: Settings
    checker: CheckerChain  # Never raises. An empty chain means the fixed rules decide alone.
    # The emergency check running alongside the main model, by conversation ID. It holds a
    # running request, not call state, so it is kept here rather than in the checkpoint.
    pending_emergency: dict[str, Any] = field(default_factory=dict)


@dataclass
class Deps:
    """Everything a call needs: the graph's objects plus speech."""

    graph: GraphDeps
    speech: Any  # An object that follows the Speech protocol.


def normalise_stop(response: AIMessage) -> Literal["complete", "cut_off"]:
    """Turn a provider's stop reason into one value.

    Providers name this differently. Anything unknown counts as cut_off, so a doubtful
    response never runs tools.
    """
    metadata = response.response_metadata
    reason = metadata.get("stop_reason") or metadata.get("finish_reason")
    return "complete" if reason in COMPLETE_STOPS else "cut_off"
