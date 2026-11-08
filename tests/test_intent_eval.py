"""Tests for the intent eval. Stage: checker design, Step 6.

What this file is:
    Proves the parts of evals/intent_eval.py the checker design's gate depends on, with no
    network.

What it contains:
    - load_cases joins case files section by section, and stops on a repeated id.
    - The product backend sends the product's questions with the right role and time limit.
    - The product row combines layers with the product's own rules, and a failed backend falls
      back to the fixed rules.
    - The time limits come from the same settings as the cut-off.
    - The system gate counts a failed backend's phrase alarm as a false alarm.

How to understand it:
    The gate is only as good as the eval behind it, so the eval must measure exactly what the
    bot does: the same questions, the same roles, the same time limits, the same combining
    rules. A failed backend must count as the fixed rules alone, as it does in the bot.

Used by:
    pytest
"""

import pytest
from evals.intent_eval import load_cases, product_backend, product_limits, product_says
from evals.system_gate import gate
from tests.fakes import ScriptedChecker

from dental_agent.checker import CheckAnswer, caller_questions, reply_questions
from dental_agent.config import Settings


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_load_cases_joins_the_sections_of_several_files(tmp_path):
    first = write(tmp_path, "a.yaml", 'emergency:\n  - {id: a1, text: "x", label: true}\n')
    second = write(
        tmp_path,
        "b.yaml",
        'emergency:\n  - {id: b1, text: "y", label: false}\n'
        '"yes":\n  - {id: b2, text: "yes", label: true}\n',
    )
    cases = load_cases([first, second])
    assert [c["id"] for c in cases["emergency"]] == ["a1", "b1"]
    assert [c["id"] for c in cases["yes"]] == ["b2"]


def test_load_cases_stops_on_a_repeated_id(tmp_path):
    first = write(tmp_path, "a.yaml", 'emergency:\n  - {id: a1, text: "x", label: true}\n')
    with pytest.raises(SystemExit, match="a1"):
        load_cases([first, first])


def answer(probabilities=None, **answers):
    return CheckAnswer(layer="jev", answers=answers, probabilities=probabilities or {}, model="m")


def test_the_product_backend_asks_the_products_questions_with_the_right_role():
    backend = ScriptedChecker(
        "jev",
        [
            answer(emergency=True),
            answer(claims_change=False, medical_advice=True),
            answer(yes=True),
        ],
    )
    ask = product_backend(backend, {"jev": 0.7})
    assert ask("emergency", "help")["emergency"] is True
    assert ask("reply", "rinse")["medical_advice"] is True
    assert ask("yes", "absolutely")["yes"] is True
    assert backend.questions[0] == caller_questions()
    assert backend.questions[1] == reply_questions()
    assert list(backend.questions[2]) == ["yes"]  # The eval-only yes question.
    assert backend.roles == ["caller", "reply", "caller"]
    assert backend.limits == [0.7] * 3


def test_the_product_backend_keeps_jevs_probabilities():
    ask = product_backend(
        ScriptedChecker("jev", [answer(probabilities={"emergency": 0.79}, emergency=True)]),
        {"jev": 1.0},
    )
    assert ask("emergency", "help")["emergency_p"] == 0.79


@pytest.mark.parametrize(
    ("kind", "fixed", "model", "expected"),
    [
        ("jev", True, False, False),  # A Jev no silences the phrase.
        ("jev", False, True, True),
        ("anthropic", True, False, True),  # With the Haiku checker either layer raises it.
        ("jev", True, None, True),  # A failed backend: the phrases decide alone.
        ("jev", False, None, False),
    ],
)
def test_the_product_emergency_row_uses_the_products_rule(kind, fixed, model, expected):
    answer_dict = {} if model is None else {"emergency": model}
    fixed_dict = {"emergency": fixed}
    assert product_says(kind, "emergency", "emergency", answer_dict, fixed_dict) is expected


def test_the_product_reply_row_raises_a_flag_when_either_layer_does():
    fixed = {"claims_change": False, "medical_advice": True}
    assert product_says("jev", "reply", "medical_advice", {}, fixed) is True
    model = {"claims_change": True, "medical_advice": False}
    assert product_says("jev", "reply", "claims_change", model, {"claims_change": False}) is True


def test_the_product_limits_come_from_the_same_settings_as_the_cutoff(monkeypatch):
    # .env and the environment change the limits the eval uses, as they do for the bot.
    monkeypatch.setenv("CHECKER_JEV_TIMEOUT_S", "0.5")
    monkeypatch.setenv("CHECKER_ANTHROPIC_TIMEOUT_S", "1.5")
    assert product_limits(Settings(_env_file=None)) == {"jev": 0.5, "anthropic": 1.5}


def test_the_gate_counts_a_failed_backends_phrase_alarm_as_a_false_alarm():
    # With no answer, the phrase raises the alarm alone, so it is not the accepted
    # Haiku-says-no case.
    cases = {
        "emergency": [{"id": "n01", "text": "I had a stroke last year", "label": False}],
        "reply": [],
    }
    rows = [{"id": "n01", "answer": {}, "error": "TimeoutError"}]  # A failed call.
    baseline = [{"id": "n01", "runs": [[], [], []]}]
    lines = gate("anthropic", rows, baseline, cases, lambda check, text: {"emergency": True})
    report = "\n".join(lines)
    assert "n01 (phrase, no checker answer)" in report
    assert "Phrase-only" not in report
