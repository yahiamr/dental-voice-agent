"""Builds what the model sees. Stage: 1.

What this file is:
    Assembles the messages for each model call.

What it contains:
    - state_block(state, actions, clock, clinic) -> str: current facts written by code,
      including who is calling and the services this caller can book.
    - build_messages(state, facts) -> list: the full message list.

How to understand it:
    The order is one system message with the prompt and state block, then messages, then
    last_caller_text, then turn_messages. The state block is rebuilt every turn from state and
    the calendar, so it is the trusted record. Putting it in the system message means code-built
    facts never look like caller words.

    After a booking the caller's appointments are reloaded, so the new one is listed under
    Found appointments as well as under actions. It is marked "booked in this call", because a
    real voice call showed the model reading the two lines as a double booking.

Uses:
    state.py, models.py, clock.py

Used by:
    graph/nodes.py (call_model)
"""

from datetime import timedelta
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from dental_agent.clinic.data import Clinic
from dental_agent.clock import Clock
from dental_agent.models import ActionRecord
from dental_agent.state import CallState

# The prompt sits next to this file, so it ships with the package.
SYSTEM_PROMPT = Path(__file__).with_name("system.md").read_text()

# How many days of dates the model sees, so it can turn "next Friday" into a date.
DATE_LIST_DAYS = 14


# How a found appointment changed in this call is marked, so it is not read as a second one.
CHANGE_MARKS = {"book": "booked in this call", "move": "moved in this call",
                "cancel": "cancelled in this call"}  # fmt: skip


def state_block(state: CallState, actions: list[ActionRecord], clock: Clock, clinic: Clinic) -> str:
    """Write the Current Facts section, rebuilt from state and the calendar every turn."""
    today = clock.now().date()
    days = [today + timedelta(days=offset) for offset in range(DATE_LIST_DAYS)]
    details = state.details
    lines = ["## Current Facts", "", f"Today: {_day(today)}", "Next 14 days:"]
    lines += [f"- {_day(day)}: {day.isoformat()}" for day in days]

    lines += [
        "",
        f"Urgent line: {clinic.urgent_line}",
        f"Clinic number: {clinic.clinic_line}",
        _caller_line(state),
        "",
        "Service IDs:",
        *[f"- {item.id}: {item.name}" for item in clinic.services],
        "",
        # Only these, so the model never offers a service the caller cannot book.
        "Services this caller can book:",
        *_bookable(state, clinic),
        "",
        "Details collected:",
        f"- name: {details.name or 'unknown'}",
        f"- service: {details.service or 'unknown'}",
        f"- preferred days: {', '.join(str(day) for day in details.preferred_days) or 'unknown'}",
        f"- time window: {details.time_window or 'unknown'}",
        "",
    ]

    # Short IDs are shown so tools can use them. Database IDs never reach the model.
    if state.offers:
        lines.append("Current offers:")
        lines += [f"- {offer.slot_id}: {offer.spoken}" for offer in state.offers]
    else:
        lines.append("Current offers: none")

    if state.found_appointments:
        # A change made in this call is marked, because after a booking the caller's
        # appointments are reloaded and the new one would otherwise look like an old one.
        changed = {action.appointment_id: CHANGE_MARKS[action.kind] for action in actions}
        lines.append("Found appointments:")
        lines += [
            f"- {item.appointment_id}: {item.service} for {item.patient_name} on {item.spoken}"
            + (f" ({changed[item.db_id]})" if item.db_id in changed else "")
            for item in state.found_appointments
        ]
    else:
        lines.append("Found appointments: none")

    if actions:
        lines.append("Actions written in this call:")
        lines += [f"- {action.result_text}" for action in actions]
    else:
        lines.append("Actions written in this call: none")

    pending = state.pending_action
    lines.append(f"Pending action: {pending.kind if pending else 'none'}")
    return "\n".join(lines)


def _caller_line(state: CallState) -> str:
    """Say who is calling. One number is one patient, so a number on record names them."""
    name = state.details.name
    if state.details.patient_type == "existing":
        return f"Caller: {name}, an existing patient (from the number's record)"
    if state.first_visit_booked:
        return f"Caller: {name}, a new patient with a first-visit consultation booked"
    if state.patient_record:
        return f"Caller: {name}, a new patient on record whose first visit has not happened"
    if name:
        return f"Caller: a new patient, who gave the name {name}"
    return "Caller: a new patient, name not given yet"


def _bookable(state: CallState, clinic: Clinic) -> list[str]:
    """List the services this caller can book. A patient is new until their first visit."""
    if state.details.patient_type == "existing":
        return [f"- {item.id}: {item.name}" for item in clinic.services if not item.first_visit]
    if state.first_visit_booked:
        return ["- none until their first visit, which is already booked"]
    return [f"- {item.id}: {item.name}" for item in clinic.services if item.first_visit]


def build_messages(state: CallState, facts: str) -> list[BaseMessage]:
    """Assemble everything the model sees for one call to it.

    facts is this turn's Current Facts text. It stays the same for every model call in a turn.
    Facts from a tool reach the model through its tool result until the next turn.
    """
    # The prompt and the facts share one system message, so facts never look like caller words.
    system = SystemMessage(SYSTEM_PROMPT + "\n\n" + facts)
    history: list[BaseMessage] = [
        HumanMessage(item["text"]) if item["role"] == "caller" else AIMessage(item["text"])
        for item in state.messages
    ]
    return [system, *history, HumanMessage(state.last_caller_text), *state.turn_messages]


def _day(day) -> str:
    """Return a date in words, such as "Monday 9 November 2026"."""
    return f"{day:%A} {day.day} {day:%B %Y}"
