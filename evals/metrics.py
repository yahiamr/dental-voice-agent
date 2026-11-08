"""One formula per metric. Stage: 4 and 5.

What this file is:
    Turns stored runs into the numbers reported in the README.

What it contains:
    - task_completion(rows, pass_at): passing runs per scenario, and whether it passes.
    - guard_rejections(trace_paths): how often each guard stepped in, by guard.
    - turn_delays(trace_paths): speech end to first audio, median and slowest, split into tool
      and no-tool turns.
    - Retrieval hit rate arrives with document search in Stage 5.

How to understand it:
    Each metric is read from trace files, so numbers can be recomputed without another run.

    check_confirmation is left out of the rejections, because it logs "reject" on every turn
    whose words are not a yes, with or without a question waiting. Counting it would bury the
    guards that really stepped in.

    A turn's delay is the time from the end of the caller's speech to the first audio of the
    reply. Text calls have no audio timings, so their turns are skipped.

Used by:
    evals/evaluate.py, tests/test_metrics.py
"""

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from statistics import median

NOT_COUNTED = {"check_confirmation"}  # Logs "reject" on every turn that is not a yes.


def task_completion(rows: Iterable[tuple[str, bool]], pass_at: int) -> dict[str, dict]:
    """Count passing runs per scenario, in the order the scenarios first appear."""
    counts: dict[str, dict] = {}
    for name, passed in rows:
        row = counts.setdefault(name, {"passed": 0, "runs": 0})
        row["runs"] += 1
        row["passed"] += int(passed)
    for row in counts.values():
        row["passed_scenario"] = row["passed"] >= pass_at
    return counts


def guard_rejections(trace_paths: Iterable[Path]) -> dict[str, int]:
    """Count rejecting guard decisions across these traces, by guard, most first."""
    counts: Counter[str] = Counter()
    for event in _events(trace_paths):
        if (
            event.get("kind") == "guard"
            and event.get("decision") == "reject"
            and event.get("name") not in NOT_COUNTED
        ):
            counts[event["name"]] += 1
    return dict(counts.most_common())


def turn_delays(trace_paths: Iterable[Path]) -> dict[str, dict]:
    """Return the turn count, median and slowest delay in seconds, for each kind of turn."""
    delays: dict[str, list[float]] = {"no-tool turns": [], "tool turns": []}
    for turn in _turns(trace_paths):
        end_at = first_audio_at = None
        for event in turn.get("events", []):
            timings = event.get("timings") or {}
            if event.get("kind") != "speech":
                continue
            if end_at is None and "speech_end_at" in timings:
                end_at = timings["speech_end_at"]
            elif end_at is not None and "first_audio_at" in timings:
                first_audio_at = timings["first_audio_at"]
                break
        if end_at is not None and first_audio_at is not None:
            kind = "tool turns" if turn.get("tools_used") else "no-tool turns"
            delays[kind].append(first_audio_at - end_at)
    return {
        kind: {
            "turns": len(values),
            "median_s": round(median(values), 2) if values else None,
            "slowest_s": round(max(values), 2) if values else None,
        }
        for kind, values in delays.items()
    }


def _turns(trace_paths: Iterable[Path]) -> Iterable[dict]:
    """Yield every turn line in these traces, skipping start lines and missing files."""
    for path in trace_paths:
        if not Path(path).is_file():
            continue
        for line in Path(path).read_text().splitlines():
            entry = json.loads(line)
            if "events" in entry:
                yield entry


def _events(trace_paths: Iterable[Path]) -> Iterable[dict]:
    """Yield every event in these traces."""
    for turn in _turns(trace_paths):
        yield from turn.get("events", [])
