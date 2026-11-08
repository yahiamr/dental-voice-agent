"""The debug log for each call. Stage: 1.

What this file is:
    One JSON line per turn, showing what happened inside it.

What it contains:
    - TraceSink, with bind, write_start, event, flush and close.

How to understand it:
    Open a trace when a call behaves oddly: it records model calls, tool calls and guard
    decisions. Events are buffered outside graph state, so a failed run keeps them. It writes
    only listed fields, never the whole Settings, so a key cannot leak. A failed trace write
    logs a warning and never breaks a call.

Used by:
    session.py, tools/dispatcher.py, graph/nodes.py, evals/metrics.py
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The only fields allowed on the start line. A list of allowed names means a key cannot reach
# the file, even if a caller passes the whole of Settings by mistake.
START_FIELDS = {
    "git_commit",
    "llm_provider",
    "llm_model",
    "speech_backend",
    "clock_time",
    "llm_timeout_s",
    "turn_deadline_s",
    "max_tool_rounds",
    "eos_wait_ms",
    "eos_wait_yes_ms",
    "silence_timeout_s",
    "min_speech_ms",
    "yes_confidence_min",
    "mic_reopen_ms",
    "speech_timeout_s",
    "checker_enabled",
    "checker_backends",
    "checker_jev_model",
    "checker_anthropic_model",
    "checker_jev_timeout_s",
    "checker_anthropic_timeout_s",
    "checker_cutoff",
}


class TraceSink:
    """Collects events during a turn, then writes them as one JSON line."""

    def __init__(self) -> None:
        self.conversation_id = ""
        self.path: Path | None = None
        # Events wait here, outside graph state, so a failed graph run keeps them.
        self.events: list[dict[str, Any]] = []

    def bind(self, conversation_id: str, runs_dir: Path) -> None:
        """Point the sink at this call's file."""
        self.conversation_id = conversation_id
        self.path = Path(runs_dir) / "traces" / f"{conversation_id}.jsonl"

    def write_start(self, fields: dict[str, Any]) -> None:
        """Write the first line, keeping only the listed fields."""
        kept = {name: value for name, value in fields.items() if name in START_FIELDS}
        self._append({"kind": "start", "conversation_id": self.conversation_id, **kept})

    def event(self, kind: str, **fields: Any) -> None:
        """Remember one thing that happened in this turn."""
        self.events.append({"kind": kind, **fields})

    def flush(self, turn: int, tools_used: bool) -> None:
        """Write this turn's events as one line, then clear them."""
        self._append(
            {
                "conversation_id": self.conversation_id,
                "turn": turn,
                "tools_used": tools_used,
                "events": self.events,
            }
        )
        self.events = []

    def close(self) -> None:
        """Write anything left over, such as events from a failed last turn."""
        if self.events:
            self.flush(turn=-1, tools_used=False)

    def _append(self, line: dict[str, Any]) -> None:
        """Add one line to the file. A failed write logs a warning and never breaks a call."""
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as file:
                # default=str keeps unusual values, such as dates, from failing the write.
                file.write(json.dumps(line, default=str) + "\n")
        except OSError as error:
            logger.warning("Could not write trace for %s: %s", self.conversation_id, error)
