"""Tests for the metrics. Stage: 4.

What this file is:
    Proves each metric formula on small sample traces.

What it contains:
    - task_completion: counts per scenario, and the 2-of-3 pass rule.
    - guard_rejections: rejections counted by guard, leaving out check_confirmation.
    - turn_delays: speech end to first audio, split into tool and no-tool turns, ignoring turns
      with no timings, such as text calls.

Used by:
    pytest
"""

import json

from evals.metrics import guard_rejections, task_completion, turn_delays


def write_trace(path, turns):
    """Write a trace file: a start line, then one line per turn."""
    lines = [{"kind": "start"}] + [
        {"conversation_id": "c", "turn": number, "tools_used": tools, "events": events}
        for number, (tools, events) in enumerate(turns, start=1)
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return path


def heard(end_at):
    return {"kind": "speech", "result": "speech", "timings": {"speech_end_at": end_at}}


def spoken(first_audio_at):
    return {"kind": "speech", "result": "spoken", "timings": {"first_audio_at": first_audio_at}}


def guard(name, decision):
    return {"kind": "guard", "name": name, "decision": decision}


def test_task_completion_counts_runs_and_applies_the_pass_rule():
    rows = [("a", True), ("a", False), ("a", True), ("b", True), ("b", False), ("b", False)]
    result = task_completion(rows, pass_at=2)
    assert result == {
        "a": {"passed": 2, "runs": 3, "passed_scenario": True},
        "b": {"passed": 1, "runs": 3, "passed_scenario": False},
    }


def test_guard_rejections_are_counted_by_guard(tmp_path):
    first = write_trace(
        tmp_path / "one.jsonl",
        [(False, [guard("check_model_reply", "reject"), guard("check_model_reply", "pass")])],
    )
    second = write_trace(
        tmp_path / "two.jsonl",
        [(True, [guard("write_to_calendar", "reject"), guard("check_model_reply", "reject")])],
    )
    assert guard_rejections([first, second]) == {"check_model_reply": 2, "write_to_calendar": 1}


def test_check_confirmation_is_left_out_because_it_rejects_every_ordinary_turn(tmp_path):
    trace = write_trace(tmp_path / "t.jsonl", [(False, [guard("check_confirmation", "reject")])])
    assert guard_rejections([trace]) == {}


def test_a_missing_trace_counts_nothing(tmp_path):
    assert guard_rejections([tmp_path / "missing.jsonl"]) == {}


def test_turn_delays_split_tool_and_no_tool_turns(tmp_path):
    trace = write_trace(
        tmp_path / "voice.jsonl",
        [
            (False, [heard(100.0), spoken(102.0)]),
            (False, [heard(200.0), spoken(203.0)]),
            (True, [heard(300.0), spoken(304.5)]),
            (False, [heard(400.0)]),  # Nothing spoken, so no delay.
        ],
    )
    result = turn_delays([trace])
    assert result["no-tool turns"] == {"turns": 2, "median_s": 2.5, "slowest_s": 3.0}
    assert result["tool turns"] == {"turns": 1, "median_s": 4.5, "slowest_s": 4.5}


def test_text_call_traces_add_no_delays(tmp_path):
    trace = write_trace(tmp_path / "text.jsonl", [(False, [{"kind": "speech", "timings": {}}])])
    assert turn_delays([trace]) == {
        "no-tool turns": {"turns": 0, "median_s": None, "slowest_s": None},
        "tool turns": {"turns": 0, "median_s": None, "slowest_s": None},
    }
