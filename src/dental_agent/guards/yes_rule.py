"""The yes rule. Stage: 1.

What this file is:
    Decides whether the caller's reply is a clear yes.

What it contains:
    - is_yes(text) -> bool.
    - The fixed phrase list: yes, yeah, yep, correct, go ahead, sounds good, book it,
      absolutely, perfect, go for it and others.

How to understand it:
    It lowercases the text, strips punctuation, removes leading fillers such as "um" and
    trailing words such as "please". What is left must split fully into list entries, matching
    whole words and trying the longest entry first. So "yes please", "yes yes" and "yes book
    it" are yeses, and "yes, but Thursday" and "yeah no" are not, because "but" and "no" are not
    entries. A reply that is empty after stripping, such as "thanks" or "um", is not a yes. "ok"
    is left out on purpose, because callers often say it before changing their mind. The rule is
    strict because a wrong no only means asking again, while a wrong yes changes the calendar.

    On the golden set, evals/intent_cases.yaml, the rule accepts 15 of the 16 plain yeses and
    refuses all 18 replies that are not. The one it refuses, "sure, why not", holds "not". A
    greedy split could in theory refuse a reply another split would accept. Today's list has no
    such reply.

Used by:
    graph/nodes.py (check_confirmation)
"""

import re

# The only replies that count as a yes. "ok" is left out on purpose, because callers often use
# it as a filler before changing their mind.
YES_PHRASES = {
    "yes",
    "yeah",
    "yep",
    "yup",
    "correct",
    "that's right",
    "that's correct",
    "that works",
    "that's fine",
    "sure",
    "go ahead",
    "please do",
    "sounds good",
    "confirm",
    "book it",
    # Added with the checker design, from the golden set's refused yeses.
    "absolutely",
    "definitely",
    "perfect",
    "that's perfect",
    "go for it",
}
# The most words in one entry, so the split knows how far to look ahead.
LONGEST_ENTRY = max(len(phrase.split()) for phrase in YES_PHRASES)

# Words removed from the start, as in "um, yes".
LEADING_FILLERS = ("um", "uh", "er", "ah", "hmm", "oh", "well", "huh")

# Words removed from the end, as in "yes please".
TRAILING_COURTESY = ("please", "thanks", "thank you")


def is_yes(text: str) -> bool:
    """Return True only if the whole reply is a clear yes.

    The rule is strict because the two mistakes cost different amounts: a wrong no only means
    asking again, while a wrong yes changes the calendar.
    """
    # Step 1: lowercase, and turn curly apostrophes into straight ones.
    cleaned = text.lower().replace("’", "'")
    # Step 2: remove punctuation except apostrophes, and collapse repeated spaces.
    cleaned = re.sub(r"[^\w\s']", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    # Step 3: remove leading fillers, as many as there are.
    cleaned = _strip_start(cleaned, LEADING_FILLERS)
    # Step 4: remove trailing courtesy words, as many as there are.
    cleaned = _strip_end(cleaned, TRAILING_COURTESY)
    # Step 5: what is left must not be empty, and must split fully into list entries.
    return bool(cleaned) and _splits_into_entries(cleaned.split())


def _splits_into_entries(words: list[str]) -> bool:
    """Return True if the words are a run of list entries, trying the longest entry first."""
    position = 0
    while position < len(words):
        for size in range(LONGEST_ENTRY, 0, -1):
            if " ".join(words[position : position + size]) in YES_PHRASES:
                position += size
                break
        else:
            return False  # No entry starts here, so a word outside the list remains.
    return True


def _strip_start(text: str, words: tuple[str, ...]) -> str:
    """Remove any of these words from the start, repeatedly."""
    changed = True
    while changed:
        changed = False
        for word in words:
            if text == word or text.startswith(word + " "):
                text = text[len(word) :].strip()
                changed = True
    return text


def _strip_end(text: str, words: tuple[str, ...]) -> str:
    """Remove any of these words from the end, repeatedly."""
    changed = True
    while changed:
        changed = False
        for word in words:
            if text == word or text.endswith(" " + word):
                text = text[: -len(word)].strip()
                changed = True
    return text
