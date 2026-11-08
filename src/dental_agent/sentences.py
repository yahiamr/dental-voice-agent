"""Fixed sentences for actions and call status. Stage: 1 and 3.

What this file is:
    The only place that writes sentences stating a successful action or the call's status.

What it contains:
    - confirmation_text: asks for a yes before a change.
    - result_text: states what was written.
    - status_text: states what has changed so far in the call.
    - closing_text: the last sentence of the call.

How to understand it:
    These are plain string templates with no LLM, so the words always match the saved data. Read
    the templates under Sentences in the design first, then each function is one if per action
    kind. Replies built here go into messages, so the model sees them on later turns but never
    rewords them.

Uses:
    models.py

Used by:
    graph/nodes.py, guards/reply_check.py, session.py
"""

# The sentence that closes a call when the calendar could not be read.
CLOSING_UNKNOWN = (
    "I couldn't check the result, so please call the clinic on {clinic_line}. "
    "If this is an emergency, call 112. Goodbye."
)


def confirmation_text(
    kind: str,
    service: str,
    spoken: str,
    name: str | None = None,
    in_addition: bool = False,
    old_spoken: str | None = None,
) -> str:
    """Return the sentence that repeats a change and asks for a yes.

    Every question says the patient's name: a booking so a misheard name can be corrected, and
    a move or cancel so the caller hears whose appointment changes. spoken is the new time for
    a book or move, and the appointment's time for a cancel. A move also needs old_spoken.
    """
    if kind == "book":
        # Warn the caller, so a yes cannot create a second appointment by accident.
        extra = ", in addition to your existing appointment" if in_addition else ""
        return f"Just to confirm: a {service} for {name} on {spoken}{extra}. Shall I book that?"
    if kind == "move":
        if not old_spoken:
            raise ValueError("A move question needs the old time")
        return (
            f"Just to confirm: move your {service} for {name} from {old_spoken} to {spoken}. "
            "Shall I go ahead?"
        )
    if kind == "cancel":
        return f"Just to confirm: cancel your {service} for {name} on {spoken}. Shall I go ahead?"
    raise ValueError(f"No confirmation question for {kind}")


def result_text(kind: str, service: str, spoken: str) -> str:
    """Return the sentence stating what was written to the calendar.

    spoken is the new time for a book or move, and the appointment's time for a cancel. None
    starts with "Done", because status_text and closing_text repeat these after "To confirm:".
    """
    if kind == "book":
        return f"You're booked for a {service} on {spoken}."
    if kind == "move":
        return f"Your {service} is now on {spoken}."
    if kind == "cancel":
        return f"Your {service} on {spoken} is cancelled."
    raise ValueError(f"No result sentence for {kind}")


def status_text(results: list[str]) -> str:
    """Return what has changed so far, given each action's result sentence."""
    if not results:
        return "Nothing has been changed on this call so far."
    return "So far on this call: " + " ".join(results)


def closing_text(results: list[str], read_failed_line: str | None = None) -> str:
    """Return the last sentence of a call.

    read_failed_line is the clinic number, passed when the calendar could not be read.
    In that case the call cannot state a result, so it sends the caller to the clinic.
    """
    if read_failed_line is not None:
        return CLOSING_UNKNOWN.format(clinic_line=read_failed_line)
    if not results:
        return "Nothing was changed on this call. Goodbye."
    return "To confirm: " + " ".join(results) + " Goodbye."
