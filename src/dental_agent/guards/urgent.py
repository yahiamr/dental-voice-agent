"""Life-threatening phrase detection. Stage: 1.

What this file is:
    Finds life-threatening phrases in the caller's words before the model runs.

What it contains:
    - detect(text, phrases, negation_tokens) -> UrgentLevel | None.
    - combine_emergency(phrase_hit, layer, checker_yes) -> bool: the checker design's rule.

How to understand it:
    It splits the text into clauses at punctuation and the word "but", then looks for each
    phrase from clinic.yaml. A match is ignored only if a negation word sits directly before it
    in the same clause. So "no chest pain" does not trigger, and "No, I can't breathe" does.
    Negation is narrow on purpose: a false alarm ends one call, while a missed emergency is
    worse. Dental pain has no phrase check. The model handles it through flag_urgent.

    combine_emergency joins the phrase result with the checker's answer:
    - Jev decides alone, so a Jev "no" can now silence a phrase match. This partly reverses the
      rule above, by the author's decision: on the golden set the phrases caused every false
      alarm and caught nothing Jev missed, and the main model's flag_urgent still ends the call.
    - The Haiku checker gives no probability, so with it either layer raises the alarm.
    - With no checker answer, the phrases decide, as before.
    It takes plain values, so guards stay free of checker types. See
    docs/designs/checker_model.md.

Uses:
    models.py

Used by:
    graph/nodes.py (check_emergency)
"""

import re
from typing import Literal

from dental_agent.models import UrgentLevel


def detect(text: str, phrases: list[str], negation_tokens: list[str]) -> UrgentLevel | None:
    """Return LIFE_THREATENING if an urgent phrase appears and is not negated, otherwise None."""
    # Step 1: lowercase, and remove apostrophes of both kinds. Callers and speech services do not
    # always write them, and "cant breathe" must match "can't breathe".
    cleaned = _plain(text)
    # Step 2: split into clauses. "No, I can't breathe" must not let "no" cancel the phrase.
    clauses = re.split(r"[,.;!?]|\bbut\b", cleaned)
    # Each phrase becomes a word list, so matching works on whole words only.
    phrase_words = [_plain(phrase).split() for phrase in phrases]
    for clause in clauses:
        # Step 3: remaining punctuation becomes a gap, because this keeps only the words.
        words = re.findall(r"[a-z0-9]+", clause)
        for wanted in phrase_words:
            for start in _match_starts(words, wanted):
                # Step 4: only the word directly before the match can negate it.
                if start == 0 or words[start - 1] not in negation_tokens:
                    # Step 5: one match that is not negated is enough.
                    return UrgentLevel.LIFE_THREATENING
    return None


def _plain(text: str) -> str:
    """Return lowercase text with straight and curly apostrophes removed."""
    return text.lower().replace("\u2019", "").replace("'", "")


def _match_starts(words: list[str], wanted: list[str]) -> list[int]:
    """Return every position where the wanted words appear in a row."""
    size = len(wanted)
    return [i for i in range(len(words) - size + 1) if words[i : i + size] == wanted]


def combine_emergency(
    phrase_hit: bool, layer: Literal["jev", "anthropic"] | None, checker_yes: bool | None
) -> bool:
    """Return whether an emergency is raised, from the phrases and the checker's answer.

    When layer is set, checker_yes is never None. A layer of None means no checker answered.
    """
    if layer == "jev":
        return bool(checker_yes)
    if layer == "anthropic":
        return phrase_hit or bool(checker_yes)
    return phrase_hit
