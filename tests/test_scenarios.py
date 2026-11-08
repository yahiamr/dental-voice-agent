"""Tests for the scenario runner. Stage: 4.

What this file is:
    Proves the scenario driver, the expectation checks, the setup steps and one whole run, with
    no API key.

What it contains:
    - ScenarioSpeech: lines in order, the automatic yes, the automatic yes to a spelled name,
      the hook that runs before a yes, and running out of lines.
    - check_run: one test per expectation key, and the lines-ran-out failure.
    - The setup steps: moving the caller's appointment, and taking the slot a question names.
    - One whole scenario run with ScriptedModel, through the real session.
    - The scenario file itself: it loads, and every scenario has a name and an expectation.

How to understand it:
    The runner must fail a run only when the bot is wrong, so these tests pin down exactly what
    each expectation checks. The real model is not used here; evals/evaluate.py runs it.

Used by:
    pytest
"""

from datetime import timedelta
from pathlib import Path

import pytest
from evals.driver import ScenarioSpeech
from evals.evaluate import (
    SCENARIOS_PATH,
    Expect,
    RunResult,
    Scenario,
    check_run,
    compare_with_baseline,
    default_out,
    is_full_run,
    load_scenarios,
    pass_needed,
    place_appointment,
    results_markdown,
    results_summary,
    run_once,
    run_settings,
    take_named_slot,
)
from tests.conftest import TEST_NOW, build_test_deps
from tests.fakes import ScriptedModel, text_reply, tool_reply

from dental_agent.config import Settings
from dental_agent.formatting import spoken_slot

LUCIA = "+34600000001"  # The first sample patient, with one check-up ahead.

# Driver


def test_lines_are_given_in_order_and_running_out_hangs_up():
    speech = ScenarioSpeech(["Hi", "Bye"])
    assert speech.transcribe(expect_yes=False).text == "Hi"
    assert speech.transcribe(expect_yes=False).text == "Bye"
    assert speech.transcribe(expect_yes=False).kind == "closed"
    assert speech.ran_out


def test_a_call_that_ends_before_the_lines_do_has_not_run_out():
    speech = ScenarioSpeech(["Hi", "Bye"])
    speech.transcribe(expect_yes=False)
    assert not speech.ran_out


def test_a_confirmation_question_is_answered_yes_without_using_a_line():
    speech = ScenarioSpeech(["Book me", "Bye"])
    speech.transcribe(expect_yes=False)
    speech.speak("Just to confirm: ... Shall I book that?", asks_confirmation=True)
    assert speech.transcribe(expect_yes=True).text == "yes"
    assert speech.transcribe(expect_yes=False).text == "Bye"


def test_without_auto_yes_the_next_line_answers_the_question():
    speech = ScenarioSpeech(["Book me", "Yes, but make it Thursday."], auto_yes=False)
    speech.transcribe(expect_yes=False)
    speech.speak("Just to confirm: ... Shall I book that?", asks_confirmation=True)
    assert speech.transcribe(expect_yes=True).text == "Yes, but make it Thursday."


def test_a_spelled_name_is_confirmed_without_using_a_line():
    # The prompt spells a new caller's name back. A fixed script cannot know where that falls.
    speech = ScenarioSpeech(["I'm Ana Ruiz", "Bye"])
    speech.transcribe(expect_yes=False)
    speech.speak("That's A-N-A, R-U-I-Z. Is that right?")
    assert speech.transcribe(expect_yes=False).text == "Yes, that's right."
    assert speech.transcribe(expect_yes=False).text == "Bye"


def test_capitals_that_are_not_a_spelling_are_not_confirmed():
    speech = ScenarioSpeech(["Hi", "Bye"])
    speech.transcribe(expect_yes=False)
    speech.speak("Is a check-up OK? Call 112 if it is an emergency.")
    assert speech.transcribe(expect_yes=False).text == "Bye"


def test_the_hook_runs_once_with_the_question_before_the_first_yes():
    seen = []
    speech = ScenarioSpeech(["Book me", "The next one", "Bye"], before_yes=seen.append)
    speech.transcribe(expect_yes=False)
    speech.speak("Question one?", asks_confirmation=True)
    speech.transcribe(expect_yes=True)
    speech.transcribe(expect_yes=False)
    speech.speak("Question two?", asks_confirmation=True)
    speech.transcribe(expect_yes=True)
    assert seen == ["Question one?"]


def test_spoken_lines_are_recorded():
    speech = ScenarioSpeech([])
    speech.speak("Hello")
    assert speech.spoken == ["Hello"]


# Expectation checks


def record(**overrides):
    """A stored conversation record that meets the default expectation below."""
    data = {
        "outcome": "completed",
        "actions": [{"kind": "book", "spoken": "Monday 16 November at 09:00"}],
        "transcript": [
            {"role": "bot", "text": "A first-visit consultation costs 30 euros."},
            {"role": "caller", "text": "Bye"},
        ],
        "details": {"name": "Ana Ruiz", "service": "first_visit"},
        "summary": {"key_points": ["Caller: Ana Ruiz", "Booked: first visit"]},
    }
    return data | overrides


def expect(**values):
    return Expect(**values)


def test_a_run_that_meets_every_expectation_passes():
    rules = expect(
        actions=["book"],
        outcome="completed",
        said=["30 euros"],
        not_said=["mg"],
        tools=["find_slots"],
        record={"details.name": "Ana Ruiz"},
        record_contains={"summary.key_points": "Ana Ruiz"},
    )
    assert check_run(rules, record(), {"find_slots"}, ran_out=False) == []


def test_the_wrong_actions_fail():
    failures = check_run(expect(actions=[]), record(), set(), ran_out=False)
    assert failures == ["actions were ['book'], expected []"]


def test_the_wrong_outcome_fails():
    failures = check_run(expect(outcome="no_action"), record(), set(), ran_out=False)
    assert failures == ["outcome was completed, expected no_action"]


def test_said_text_is_looked_for_in_the_bot_lines_only_in_any_case():
    assert check_run(expect(said=["30 EUROS"]), record(), set(), ran_out=False) == []
    failures = check_run(expect(said=["Bye"]), record(), set(), ran_out=False)
    assert failures == ["the bot never said 'Bye'"]


def test_forbidden_text_fails():
    failures = check_run(expect(not_said=["costs"]), record(), set(), ran_out=False)
    assert failures == ["the bot said 'costs'"]


def test_a_missing_tool_fails():
    failures = check_run(expect(tools=["flag_urgent"]), record(), {"find_slots"}, ran_out=False)
    assert failures == ["tool flag_urgent was never called"]


def test_a_record_field_is_read_by_its_dotted_path():
    failures = check_run(expect(record={"details.service": "cleaning"}), record(), set(), False)
    assert failures == ["record details.service was 'first_visit', expected 'cleaning'"]


def test_record_contains_looks_inside_a_list():
    failures = check_run(
        expect(record_contains={"summary.key_points": "Lucía"}), record(), set(), False
    )
    assert failures == ["record summary.key_points does not contain 'Lucía'"]


def test_running_out_of_lines_fails_even_when_the_rest_holds():
    assert check_run(expect(), record(), set(), ran_out=True) == ["the caller's lines ran out"]


@pytest.mark.parametrize(("runs", "needed"), [(1, 1), (2, 2), (3, 2), (5, 3)])
def test_a_scenario_needs_a_majority_of_its_runs(runs, needed):
    # The design's rule is 2 of 3. A single shake-out run must still be able to pass.
    assert pass_needed(runs) == needed


# Setup steps


def test_place_appointment_moves_the_callers_appointment(calendar):
    place_appointment(calendar, LUCIA, hours_ahead=6, now=TEST_NOW)
    [appointment] = calendar.appointments_for(LUCIA, 5)
    assert appointment.start_at == TEST_NOW + timedelta(hours=6)


def test_take_named_slot_books_the_slot_in_the_question_for_someone_else(calendar, clinic):
    slot = calendar.free_slots(
        TEST_NOW.date() + timedelta(days=7), TEST_NOW.date() + timedelta(days=7), None, None, 1,
        TEST_NOW,
    )[0]  # fmt: skip
    question = f"Just to confirm: a cleaning for Ana Ruiz on {spoken_slot(slot.start_at)}?"
    assert take_named_slot(calendar, question, TEST_NOW, clinic)
    assert not calendar.slot_is_free(slot.db_slot_id)


def test_take_named_slot_does_nothing_when_no_slot_is_named(calendar, clinic):
    assert not take_named_slot(calendar, "Shall I go ahead?", TEST_NOW, clinic)


# One whole run


def test_a_scripted_booking_scenario_passes_through_the_real_session(
    tmp_path, clinic, clock, calendar
):
    scenario = Scenario(
        name="scripted_booking",
        stage=4,
        caller="new",
        lines=["Book me a first visit", "Bye"],
        expect=Expect(actions=["book"], outcome="completed", tools=["prepare_booking"]),
    )
    script = [
        tool_reply(
            ("save_details", {"name": "Ana Ruiz", "service": "first_visit"}),
            (
                "find_slots",
                {"service": "first_visit", "date_from": "2026-11-16", "date_to": "2026-11-20"},
            ),
        ),
        tool_reply(("prepare_booking", {"slot_id": "s1"})),
        tool_reply(("end_call", {})),
        text_reply("Summary."),
    ]
    deps = build_test_deps(tmp_path, clinic, clock, calendar, model=ScriptedModel(script=script))
    result = run_once(scenario, deps)
    assert result.failures == []
    assert result.passed
    assert result.conversation_path.exists()


# The scenario file


def test_the_scenario_file_loads_with_unique_names():
    scenarios = load_scenarios(SCENARIOS_PATH)
    names = [item.name for item in scenarios]
    assert len(names) == len(set(names))
    assert len(scenarios) == 19


@pytest.mark.parametrize("scenario", load_scenarios(SCENARIOS_PATH), ids=lambda item: item.name)
def test_every_scenario_has_lines_and_checks_something(scenario):
    assert scenario.lines
    assert scenario.expect.model_dump(exclude_defaults=True)


# Saved results and the baseline


def result(name, passed, failures=()):
    return RunResult(name, 0, passed, list(failures), Path("c.json"), Path("t.jsonl"))


def scenario(name, stage=4):
    return Scenario(name=name, stage=stage, lines=["Hi"], expect=Expect(outcome="no_action"))


def test_the_summary_holds_each_scenarios_counts_and_failures():
    settings = Settings(_env_file=None)
    results = [result("a", True), result("a", False, ["the bot said 'mg'"]), result("a", True)]
    summary = results_summary([scenario("a")], results, settings, runs=3)
    assert summary["model"] == f"{settings.llm_provider}:{settings.llm_model}"
    assert summary["runs"] == 3 and summary["pass_needed"] == 2
    assert summary["scenarios"]["a"] == {
        "stage": 4,
        "passed": 2,
        "runs": 3,
        "passed_scenario": True,
        "failures": ["the bot said 'mg'"],
    }
    assert summary["passed_scenarios"] == 1 and summary["total_scenarios"] == 1


def summary_of(**passes):
    """A saved summary where each keyword is a scenario and its passing runs out of 3."""
    return {
        "scenarios": {
            name: {"passed": count, "runs": 3, "passed_scenario": count >= 2}
            for name, count in passes.items()
        }
    }


def test_a_scenario_that_stopped_passing_is_a_regression():
    lines = compare_with_baseline(summary_of(a=1, b=3), summary_of(a=3, b=3))
    assert lines == ["Regression: `a` passed 3 of 3 in the baseline, and 1 of 3 now."]


def test_fixes_and_new_or_removed_scenarios_are_listed():
    lines = compare_with_baseline(summary_of(a=3, new=3), summary_of(a=0, gone=3))
    assert lines == [
        "Fixed: `a` passed 0 of 3 in the baseline, and 3 of 3 now.",
        "New: `new`, 3 of 3.",
        "Removed: `gone`.",
    ]


def test_no_change_says_so():
    assert compare_with_baseline(summary_of(a=3), summary_of(a=2)) == [
        "No scenario changed between pass and fail."
    ]


def test_only_a_full_run_may_replace_the_saved_baseline():
    assert is_full_run(scenario_filter=[], runs=3)
    assert not is_full_run(scenario_filter=["cancel"], runs=3)
    assert not is_full_run(scenario_filter=[], runs=1)


# Review fixes


def test_no_said_text_is_a_short_number_that_any_time_would_match():
    # "30" matched "09:30", so a price answer was never really checked.
    for item in load_scenarios(SCENARIOS_PATH):
        for text in item.expect.said:
            assert not (text.isdigit() and len(text) < 3), (item.name, text)


def test_the_product_model_with_the_checker_saves_to_the_main_baseline():
    assert default_out(Settings(_env_file=None)).name == "scenarios"


def test_another_model_or_no_checker_saves_to_its_own_file():
    other = Settings(_env_file=None, llm_provider="openrouter", llm_model="vendor/model-1")
    assert default_out(other).name == "scenarios-openrouter-vendor_model-1"
    no_checker = Settings(_env_file=None, checker_enabled=False)
    assert default_out(no_checker).name.endswith("-no-checker")


def test_each_run_keeps_its_calendar_and_checkpoints_in_its_own_folder(tmp_path):
    settings = run_settings(Settings(_env_file=None, checkpointer="sqlite"), tmp_path / "run")
    assert settings.checkpointer == "memory"
    assert settings.calendar_path.parent == tmp_path / "run"
    assert settings.runs_dir == tmp_path / "run"


def test_failure_text_cannot_break_the_results_table():
    summary = {
        "model": "m", "checker": True, "commit": "c", "date": "d", "runs": 3, "pass_needed": 2,
        "passed_scenarios": 0, "total_scenarios": 1, "guard_rejections": {},
        "scenarios": {"a": {"stage": 4, "passed": 0, "runs": 3, "passed_scenario": False,
                            "failures": ["the run crashed: bad | value\nsecond line"]}},
    }  # fmt: skip
    row = next(line for line in results_markdown(summary).splitlines() if line.startswith("| `a`"))
    assert row.count(" | ") == 4
    assert "bad \\| value second line" in row


def test_place_appointment_refuses_a_slot_another_patient_holds(calendar):
    # Carmen Vidal's check-up is on Thursday 19 November at 11:30 at the fixed clock.
    carmen = calendar.appointments_for("+34600000003", 5)[0]
    hours = (carmen.start_at - TEST_NOW).total_seconds() / 3600
    with pytest.raises(ValueError, match="already taken"):
        place_appointment(calendar, LUCIA, hours_ahead=hours, now=TEST_NOW)
