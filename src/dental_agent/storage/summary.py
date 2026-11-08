"""The call summary. Stage: 1.

What this file is:
    The key points the assignment asks for.

What it contains:
    - key_points(state, actions) -> list[str]: built by code.
    - optional_note(model, key_points, deadline_s) -> str | None: one sentence from the model.

How to understand it:
    The key points come from state and the calendar, so they are true even if the model fails.
    The model sentence is an extra. If that call fails, only the sentence is lost.

Uses:
    state.py, models.py

Used by:
    session.py
"""

import logging
import re
from typing import Any

from dental_agent.models import ActionRecord
from dental_agent.state import CallState

logger = logging.getLogger(__name__)

NOTE_PROMPT = (
    "Summarise this dental call in one sentence, using only these points. "
    "If the points include an urgent flag, that urgent flag is about the caller's health, not "
    "about any service or booking, so state it as a separate fact. Mention nothing that is not "
    "in the points. Reply with that sentence and nothing else."
)


def key_points(state: CallState, actions: list[ActionRecord]) -> list[str]:
    """Write the key points from state and the calendar, with no model involved."""
    details = state.details
    points: list[str] = []
    # Add only what the caller actually gave, so a short call still has a true summary.
    if details.name:
        points.append(f"Caller: {details.name}")
    if details.patient_type:
        points.append(f"Patient type: {details.patient_type}")
    if details.service:
        points.append(f"Service: {details.service}")
    if details.preferred_days:
        days = ", ".join(day.isoformat() for day in details.preferred_days)
        points.append(f"Preferred days: {days}")
    if details.time_window:
        points.append(f"Time window: {details.time_window}")
    if state.urgent_level:
        points.append(f"Urgent: {state.urgent_level}")
    # The result sentences come from the calendar, so these are true by construction.
    points += [action.result_text for action in actions]
    if not actions:
        points.append("No changes were made.")
    return points


def optional_note(model: Any, points: list[str], deadline_s: float) -> str | None:
    """Ask the model for one summary sentence. A failure loses only this sentence."""
    prompt = NOTE_PROMPT + "\n\n" + "\n".join(points)
    try:
        response = model.bind(timeout=deadline_s).invoke(prompt)
    except Exception as error:  # Any failure here must not affect the stored call.
        logger.warning("Could not get a summary sentence: %s", error)
        return None
    return _first_sentence(response.text) or None


def _first_sentence(text: str) -> str:
    """Keep one sentence, so the model cannot add comments to a stored record."""
    return re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
