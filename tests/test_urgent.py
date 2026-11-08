"""Tests for urgent detection. Stage: 1.

What this file is:
    Proves emergencies trigger and normal dental talk does not.

What it contains:
    - Must-trigger and must-not-trigger cases, punctuated and unpunctuated.
    - About 30 normal dental sentences.
    - combine_emergency for every checker layer and answer.

Used by:
    pytest
"""

import pytest

from dental_agent.guards.urgent import combine_emergency, detect
from dental_agent.models import UrgentLevel

LIFE_THREATENING = UrgentLevel.LIFE_THREATENING

MUST_TRIGGER = [
    "my son can't breathe",
    "no pain but my throat is swelling and I can't breathe",
    "the bleeding won't stop",
    "No, I can't breathe",
    "no I can't breathe",
    "No. He's choking.",
    "no he's choking",
    "No, my throat is swelling",
    "I don't feel well, can't breathe",
    "I've never had chest pain like this",
    "I'm not really short of breath",  # Negation stays narrow, so this triggers on purpose.
    "my mother can’t breathe",  # A curly apostrophe.
    "i cant breathe and my face turned blue",  # No apostrophe, from a real call.
    "the bleeding wont stop",
    "it wont stop bleeding",
]

MUST_NOT_TRIGGER = [
    "no swelling",
    "no chest pain",
    "the bleeding stopped",
    "I'm not short of breath",
    "I've had a bit of toothache",
]

NORMAL_DENTAL_SENTENCES = [
    "I'd like to book a check-up",
    "can I move my cleaning to Thursday",
    "how much is a filling",
    "my tooth hurts when I drink something cold",
    "I need to cancel my appointment",
    "do you have anything next week in the morning",
    "I've got a bit of a toothache",
    "my gums bleed a little when I brush",
    "I lost a filling yesterday",
    "what time do you open on Friday",
    "I'm a new patient",
    "my name is Ana Ruiz",
    "is whitening safe",
    "do you take insurance",
    "where can I park",
    "can I bring my daughter",
    "I would prefer the afternoon",
    "I want a cleaning",
    "tuesday works for me",
    "the first one please",
    "yes that's right",
    "no thanks",
    "that's all for today",
    "my wisdom tooth is sore",
    "I chipped a tooth",
    "my crown feels loose",
    "I have sensitive teeth",
    "what do you charge for a first visit",
    "can you repeat that",
    "goodbye",
]


@pytest.mark.parametrize("text", MUST_TRIGGER)
def test_emergencies_trigger(text, clinic):
    assert detect(text, clinic.urgent_phrases, clinic.negation_tokens) is LIFE_THREATENING


@pytest.mark.parametrize("text", MUST_TRIGGER)
def test_emergencies_trigger_without_punctuation(text, clinic):
    # Speech services do not always add punctuation.
    bare = text.replace(",", "").replace(".", "")
    assert detect(bare, clinic.urgent_phrases, clinic.negation_tokens) is LIFE_THREATENING


@pytest.mark.parametrize("text", MUST_NOT_TRIGGER + NORMAL_DENTAL_SENTENCES)
def test_normal_sentences_do_not_trigger(text, clinic):
    assert detect(text, clinic.urgent_phrases, clinic.negation_tokens) is None


def test_a_phrase_inside_a_longer_word_does_not_trigger(clinic):
    # "stroke" must match a whole word, not part of "brushstroke".
    assert detect("the brushstroke", clinic.urgent_phrases, clinic.negation_tokens) is None


@pytest.mark.parametrize(
    ("phrase_hit", "layer", "checker_yes", "expected"),
    [
        # Jev decides alone, so its "no" silences a phrase and its "yes" needs no phrase.
        (True, "jev", True, True),
        (True, "jev", False, False),
        (False, "jev", True, True),
        (False, "jev", False, False),
        # The Haiku checker gives no probability, so either layer raises the alarm.
        (True, "anthropic", True, True),
        (True, "anthropic", False, True),
        (False, "anthropic", True, True),
        (False, "anthropic", False, False),
        # No checker answered, so the phrases decide, as on main.
        (True, None, None, True),
        (False, None, None, False),
    ],
)
def test_combine_emergency(phrase_hit, layer, checker_yes, expected):
    assert combine_emergency(phrase_hit, layer, checker_yes) is expected
