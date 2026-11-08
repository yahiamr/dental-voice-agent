"""Tests for the yes rule. Stage: 1.

What this file is:
    Proves is_yes accepts clear yeses and rejects everything else.

What it contains:
    - Every row of the True and False table under Guard Rules.
    - All 34 yes-group cases from the golden set, evals/intent_cases.yaml. y15, "sure, why not",
      is expected to be refused, because the rule refuses any word outside its list.
    - Replies that are only fillers or courtesy words, which must be refused.

Used by:
    pytest
"""

import pytest
import yaml
from tests.conftest import PROJECT_ROOT

from dental_agent.guards.yes_rule import is_yes

GOLDEN = yaml.safe_load((PROJECT_ROOT / "evals" / "intent_cases.yaml").read_text())["yes"]
# The one plain yes the rule refuses on purpose: "not" is outside the list.
REFUSED_ON_PURPOSE = {"y15"}

TRUE_ROWS = [
    "Yes.",
    "yes please",
    "Yes, thank you.",
    "um, yes",
    "oh yeah",
    "Yep!",
    "that's right",
    "that's correct please",
    "that works",
    "that's fine",
    "sure",
    "go ahead",
    "please do",
    "sounds good",
    "confirm",
    "book it thanks",
    "That’s right",  # A curly apostrophe, as speech services often return.
]

FALSE_ROWS = [
    "ok",
    "okay then",
    "no",
    "no thanks",
    "no, that's fine",
    "yes, but Thursday",
    "yes but make it Thursday",
    "yeah no",
    "wait",
    "actually, no",
    "sure, what about Thursday",
    "yes, Tuesday at ten",
    "yes, I'd rather change it",
    "I think so",
    "maybe",
    "sounds great",
    "hello",
    "",
]


@pytest.mark.parametrize("text", TRUE_ROWS)
def test_clear_replies_count_as_yes(text):
    assert is_yes(text)


@pytest.mark.parametrize("text", FALSE_ROWS)
def test_anything_else_does_not_count_as_yes(text):
    assert not is_yes(text)


@pytest.mark.parametrize("case", GOLDEN, ids=[case["id"] for case in GOLDEN])
def test_the_golden_yes_cases(case):
    expected = case["label"] and case["id"] not in REFUSED_ON_PURPOSE
    assert is_yes(case["text"]) is expected


@pytest.mark.parametrize("text", ["thanks", "please", "um", "huh", "um, thanks"])
def test_a_reply_of_only_fillers_or_courtesy_is_not_a_yes(text):
    assert not is_yes(text)
