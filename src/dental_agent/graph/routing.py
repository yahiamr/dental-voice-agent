"""Decides the next graph step. Stage: 1.

What this file is:
    The conditional edges: small functions that look at state and name the next node.

What it contains:
    - A routing function after each step that has more than one exit: check_emergency,
      check_confirmation, write_to_calendar, call_model and run_tools.

How to understand it:
    These are pure functions, so each can be tested with a state object and no graph. They never
    change anything. For example, after run_tools, a successful prepare goes to ask_confirmation,
    and more work under 3 rounds goes back to call_model.

Uses:
    state.py, models.py

Used by:
    graph/build.py
"""

from dental_agent.models import ToolExit
from dental_agent.state import CallState


def after_check_emergency(state: CallState) -> str:
    """A life-threatening phrase goes straight to the emergency line."""
    return "use_fixed_line" if state.fixed_reason else "check_confirmation"


def after_check_confirmation(state: CallState) -> str:
    """Write only after a clear yes. check_confirmation deletes the action on anything else."""
    return "write_to_calendar" if state.pending_action else "call_model"


def after_write_to_calendar(state: CallState) -> str:
    """A written action is finished. A refused or failed one needs its fixed line."""
    return "use_fixed_line" if state.fixed_reason else "end_turn"


def after_call_model(state: CallState) -> str:
    """Stop on a failure, run any tool calls, or check the model's final reply."""
    if state.fixed_reason:
        return "use_fixed_line"
    # The last working message is the response call_model just added.
    return "run_tools" if state.turn_messages[-1].tool_calls else "check_model_reply"


def after_run_tools(state: CallState) -> str:
    """Decide what follows a tool round, from the exit the dispatcher chose."""
    if state.fixed_reason:
        return "use_fixed_line"
    exits = {
        ToolExit.PREPARED: "ask_confirmation",
        ToolExit.END_CALL: "end_turn",
        ToolExit.URGENT: "use_fixed_line",
        ToolExit.ROUND_LIMIT: "use_fixed_line",
        ToolExit.NONE: "call_model",
    }
    return exits[state.tool_exit]
