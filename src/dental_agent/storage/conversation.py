"""The conversation JSON file. Stage: 1.

What this file is:
    The assignment's JSON storage: one file per call.

What it contains:
    - ConversationRecord: transcript, details, actions, urgent level, outcome, end reason and
      summary.

How to understand it:
    It is written to runs/conversations/<id>.json after each turn. The code writes a temporary
    file and then swaps it in, so a crash leaves the last complete file. transcript holds
    everything said, including lines the model never sees.

Uses:
    models.py

Used by:
    session.py
"""

import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from dental_agent.models import Details, EndReason, Outcome, UrgentLevel

logger = logging.getLogger(__name__)


class Summary(BaseModel):
    """The call summary. key_points come from code, and note is one optional model sentence."""

    key_points: list[str] = Field(default_factory=list)
    note: str | None = None


class ConversationRecord(BaseModel):
    """Everything stored about one call, in the format under Record Formats."""

    conversation_id: str
    started_at: str
    ended_at: str | None = None
    caller_phone: str
    # Everything said, including lines the model never sees, such as a repeated question.
    transcript: list[dict[str, Any]] = Field(default_factory=list)
    details: Details = Field(default_factory=Details)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    urgent_level: UrgentLevel | None = None
    outcome: Outcome | None = None
    end_reason: EndReason | None = None
    summary: Summary = Field(default_factory=Summary)

    def save(self, runs_dir: Path) -> None:
        """Write the file after each turn, so a crash leaves the last complete version.

        The text goes to a temporary file first, and os.replace swaps it in as one step. A
        reader sees the old file or the new one, never half of one. A failed save logs a
        warning and does not raise, because the caller is still on the line.
        """
        folder = Path(runs_dir) / "conversations"
        target = folder / f"{self.conversation_id}.json"
        temporary = target.with_suffix(".tmp")
        try:
            folder.mkdir(parents=True, exist_ok=True)
            temporary.write_text(self.model_dump_json(indent=2))
            os.replace(temporary, target)
        except OSError as error:
            logger.warning("Could not save conversation %s: %s", self.conversation_id, error)
            # Do not leave the temporary file behind.
            temporary.unlink(missing_ok=True)
