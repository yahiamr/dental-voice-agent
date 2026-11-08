"""Checks the model's text before it is spoken. Stage: 1, 3 and 5.

What this file is:
    The last check on anything the model wrote.

What it contains:
    - check_reply(text, actions, found_appointments, clinic, max_sentences, model_claim,
      model_advice, sources) -> ReplyDecision.
    - check_lead_in(text, service_id, actions, found_appointments, clinic, model_claim,
      model_advice, sources) -> ReplyDecision.
    - repeats_sources(text, sources): whether every statement repeats the clinic's own text.
      The checker's advice flag alone does not replace such a reply (reason "sourced"): the
      handbook's aftercare text is the clinic's, not the model's. Each statement, including any
      question but a plain offer of more help, needs 75% of its meaningful words in one source
      sentence or pair, every number, no order word swapped for its opposite ("after" for
      "before"), the same negations, and a shared pair of neighbouring words in every clause,
      so an instruction cannot ride on a long handbook sentence. A dosing word is never
      excused, and claims are still checked.
    - combine_reply(phrase_claim, dosing_hit, model_claim, model_advice): whether a claim and
      whether advice is flagged, by either layer. The eval uses it too.
    - claimed_kinds(text, claim_phrases): the kinds of change the text claims, leaving out a
      phrase that an offer leads to. The eval's fixed rules use it too.
    - fix_phone_numbers(text, numbers) -> (text, fixed): writes the clinic's numbers exactly.

How to understand it:
    check_reply runs five steps: remove markdown and IDs, correct the clinic's phone numbers,
    replace the reply with the medical line if it has dosing words, check claims, then trim to
    3 sentences. A claim is a phrase such as "you're booked", unless an offer that opens its
    clause leads straight to it, as in "shall I get you all set?". The reply must also contain
    the spoken text of a real action of that kind. A booking claim may name a found appointment
    instead, as the caller already has it. A move or cancel claim may not: a false one names the
    appointment's own time, so only a real move or cancel proves it. If the proof is missing,
    the reply becomes status_text. That is why spoken strings must be exact.

    A caller given a wrong number cannot reach the clinic, so phone numbers are fixed by code.
    fix_phone_numbers finds runs of 6 or more digits, which may be split by spaces, hyphens or
    dots. A run within 2 digit edits of the clinic line or the urgent line, such as "910 000 1",
    is rewritten as that number exactly. Shorter numbers, such as prices, times and 112, and
    numbers far from the clinic's, such as the caller's own, are left alone. A corrected reply
    still passes, with reason "phone", so a lead-in keeps its sentence and the trace shows the
    fix.

    check_lead_in is stricter, because its sentence is spoken just before a confirmation
    question. It runs check_reply, then drops the text if it asks anything, names a day, time or
    date, or names a service other than the one being booked. Those rules mean the model's
    sentence can never disagree with the question, or change what a yes answers. A dropped
    lead-in becomes empty text, so the caller hears only the question, as before.

    model_claim and model_advice are the checker's answers about the same text. Both default to
    false, which gives the result on main. With them, and after the phone numbers are
    corrected, check_reply decides in this order:
    1. Advice: a dosing word or model_advice gives the medical line. The reason is dosing if a
       word matched, otherwise model_advice.
    2. Claim phrases, kind by kind, each needing proof of the same kind, or a found appointment
       for a booking claim. The reason is claim.
    3. model_claim with no claim phrase needs a real record of any kind: the spoken time of any
       action in this call, or of a found appointment, in the reply. Otherwise the reply becomes
       the status line, with the reason model_claim. A checker claim has no kind, so any record
       is accepted. That is an accepted limit in the checker design.
    4. Otherwise, trim. The reason is phone if a number was corrected, otherwise none.
    The flags are plain booleans, so guards stay free of checker types.

Uses:
    models.py, sentences.py

Used by:
    graph/nodes.py (check_model_reply and ask_confirmation)
"""

import re
from collections.abc import Sequence
from itertools import pairwise

from dental_agent.clinic.data import Clinic, Service
from dental_agent.clinic.retriever import FILLER_WORDS
from dental_agent.models import ActionRecord, Appointment, ReplyDecision
from dental_agent.sentences import status_text

# Words and patterns that say when. Only the code's question may say when. "May" is left out,
# because "you may" is common and a lead-in naming May would still need a date or time.
WHEN_WORDS = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december",
    "today", "tomorrow", "tonight", "noon", "midnight",
)  # fmt: skip
# A claim phrase is not a claim when an offer opens its clause and leads straight to it, with
# at most OFFER_GAP words between, as in "shall I get you all set?". A question alone is not
# enough: "you're booked for Tuesday - anything else?" still claims a booking.
CLAUSE_BREAK = re.compile(r"(?<=[.!?,;])\s+")
OFFER_OPENERS = (
    "shall i", "should i", "can i", "could i", "may i", "would you like me to",
    "would you like", "do you want me to", "do you want", "want me to", "are you", "is that",
    "is it",
)  # fmt: skip
OFFER_GAP = 3
WHEN_PATTERNS = (
    r"\b\d{1,2}[:.]\d{2}\b",  # 9:00 or 15.30
    r"\b\d{1,2}\s*(am|pm)\b",  # 9am or 3 pm
    r"\bo'clock\b",
)

# A run of digits that may be split by single spaces, hyphens or dots, such as 910 000 001.
DIGIT_RUN = re.compile(r"\d(?:[ .-]?\d)+")
MIN_PHONE_DIGITS = 6
MAX_PHONE_EDITS = 2


# How much of a statement must be in one search result for it to count as repeated.
SOURCE_OVERLAP = 0.75
# Words that flip a meaning, so a repeated statement must keep every one of them.
NEGATIONS = frozenset({"not", "no", "never", "cannot", "without", "nothing", "nobody"})
# Numbers as words, which a repeated statement must also keep exactly.
_NUMBER_TEXT = """
one two three four five six seven eight nine ten eleven twelve fifteen twenty thirty forty fifty
sixty hundred half
"""
NUMBER_WORDS = frozenset(_NUMBER_TEXT.split())
# Words of order or amount, with their opposites. Search drops some of them as filler, but
# "before eating" and "after eating" mean different things, so a statement may not use one
# where its source has only the opposite. "after a local anaesthetic" for "if you had a local
# anaesthetic" is still fine.
OPPOSITES = {"before": "after", "more": "less", "most": "least"}
OPPOSITES |= {second: first for first, second in OPPOSITES.items()}
ORDER_WORDS = frozenset(OPPOSITES)
# The only words a question may have and still be skipped, such as "Is there anything else I
# can help with?". Any other question must repeat the sources too, so advice cannot hide in one.
OFFER_WORDS = frozenset({"anything", "else", "help", "today", "other", "something", "question"})


def combine_reply(
    phrase_claim: bool, dosing_hit: bool, model_claim: bool, model_advice: bool
) -> tuple[bool, bool]:
    """Return (claim flagged, advice flagged). Either layer can raise each flag."""
    return phrase_claim or model_claim, dosing_hit or model_advice


def check_reply(
    text: str,
    actions: list[ActionRecord],
    found_appointments: list[Appointment],
    clinic: Clinic,
    max_sentences: int,
    model_claim: bool = False,
    model_advice: bool = False,
    sources: Sequence[str] = (),
) -> ReplyDecision:
    """Check the model's reply, and return the text that is safe to speak.

    model_claim and model_advice are the checker's answers. Both false gives the result on main.
    sources are the texts search returned this turn. A reply that only repeats them is the
    clinic's own written text, so the checker's advice flag alone does not replace it.
    """
    # Step 1: clean text meant for a screen. Markdown and IDs must never be read aloud.
    cleaned = re.sub(r"[*_#`]", "", text)
    cleaned = re.sub(r"\b[sa]\d+\b", "", cleaned).strip()

    # Step 2: a shortened or changed clinic number becomes the exact one.
    cleaned, fixed = fix_phone_numbers(cleaned, [clinic.clinic_line, clinic.urgent_line])
    lowered = cleaned.lower()
    dosing_hit = _has_word(cleaned, clinic.dosing_words)
    matched = claimed_kinds(cleaned, clinic.claim_phrases)
    _, advice = combine_reply(bool(matched), dosing_hit, model_claim, model_advice)

    # Step 3: medical advice is never allowed, so advice from either layer replaces the reply.
    # The one exception: the checker's flag alone, on a reply that only repeats the clinic's
    # own handbook. A dosing word is never excused, whatever the source says.
    excused = advice and not dosing_hit and repeats_sources(cleaned, sources)
    if advice and not excused:
        reason = "dosing" if dosing_hit else "model_advice"
        return ReplyDecision(text=clinic.line("medical"), replaced=True, reason=reason)

    # Step 4: any claim of a change must be backed by a real record.
    results = [item.result_text for item in actions]
    found_spoken = [item.spoken.lower() for item in found_appointments if item.spoken]
    for kind in matched:
        # The reply must contain the exact time of a same-kind action. A booking claim may name
        # an appointment the caller already has. A false move or cancel names the found
        # appointment's own time, so for those only a real record is proof.
        proof = [item.spoken.lower() for item in actions if item.kind == kind]
        if kind == "book":
            proof += found_spoken
        if not any(spoken in lowered for spoken in proof):
            return ReplyDecision(text=status_text(results), replaced=True, reason="claim")

    # Step 4b: a claim only the checker saw has no kind, so a record of any kind is the proof.
    if model_claim and not matched:
        proof = [item.spoken.lower() for item in actions if item.spoken] + found_spoken
        if not any(spoken in lowered for spoken in proof):
            return ReplyDecision(text=status_text(results), replaced=True, reason="model_claim")

    # Step 5: trim, because long replies are hard to follow by ear.
    reason = "sourced" if excused else "phone" if fixed else None
    return ReplyDecision(text=_trim(cleaned, max_sentences), replaced=False, reason=reason)


def repeats_sources(text: str, sources: Sequence[str]) -> bool:
    """Return True if every statement in the text repeats the search results.

    A statement is any sentence except an offer such as "Is there anything else I can help
    with?", a question made only of OFFER_WORDS. It repeats a source sentence, or two
    neighbouring ones, when at least SOURCE_OVERLAP of its meaningful words are there, every
    number in it is there, no order word stands where the source has only its opposite, and it
    has exactly the same negations, so "two to three hours" cannot become "five hours",
    "before eating" cannot become "after eating", and "take care not to bite" cannot lose its
    "not". One statement that repeats nothing makes the
    whole reply the model's own.
    """
    statements = [item for item in _sentences(text) if not _is_offer(item)]
    if not statements or not sources:
        return False
    # Each sentence, and each pair of neighbouring sentences, since a reply may join two.
    windows: list[list[str]] = []
    for source in sources:
        sentences = [_words(item) for item in _sentences(source)]
        windows += sentences + [a + b for a, b in pairwise(sentences)]
    return all(_repeats(statement, windows) for statement in statements)


def _sentences(text: str) -> list[str]:
    """Split text into sentences at full stops, question marks and exclamation marks."""
    return [item for item in re.split(r"(?<=[.!?])\s+", text.strip()) if item]


def _is_offer(sentence: str) -> bool:
    """Return True for a question that offers more help and says nothing of its own."""
    return sentence.rstrip().endswith("?") and set(_meaningful(sentence)) <= OFFER_WORDS


def _meaningful(text: str) -> list[str]:
    """Return the words that carry meaning: all but filler, keeping negations and order words."""
    kept = NEGATIONS | ORDER_WORDS
    return [word for word in _words(text) if word not in FILLER_WORDS or word in kept]


def _repeats(statement: str, windows: list[list[str]]) -> bool:
    """Return True if this one statement repeats one window of source sentences.

    Besides the overlap, numbers, order words and negations, every clause of two or more
    meaningful words must share a pair of neighbouring words with the window. Word overlap
    alone let an instruction ride on a long handbook sentence ("…before eating anything hot,
    and hold ice on it"), because most of the sentence matched. A paraphrase keeps some of the
    handbook's word pairs ("not bite", "two three"), and an added instruction has none.
    """
    clauses = _clauses(statement)
    words = _meaningful(statement)
    if not words:
        return True  # Nothing meaningful, such as "OK.", says nothing of its own.
    numbers = {word for word in words if word.isdigit() or word in NUMBER_WORDS}
    orders = {word for word in words if word in ORDER_WORDS}
    negations = {word for word in words if word in NEGATIONS}
    for window in windows:
        available = set(window)
        same_negations = negations == {word for word in window if word in NEGATIONS}
        # An order word the source lacks, where the source has its opposite, flips the meaning.
        swapped = any(word not in available and OPPOSITES[word] in available for word in orders)
        shared = sum(word in available for word in words)
        if (
            same_negations
            and not swapped
            and numbers <= available
            and shared >= SOURCE_OVERLAP * len(words)
            and all(clause & _pairs(window) for clause in clauses)
        ):
            return True
    return False


# Where a statement splits into clauses, so an added instruction is checked on its own.
CLAUSE_BREAK = re.compile(r"[,;:()]|\b(?:and|so|but|then|or|while)\b")


def _clauses(statement: str) -> list[set[tuple[str, str]]]:
    """Return the word pairs of each clause with two or more meaningful words."""
    pairs = [_pairs(_words(part)) for part in CLAUSE_BREAK.split(statement)]
    return [item for item in pairs if item]


def _pairs(words: list[str]) -> set[tuple[str, str]]:
    """Return the neighbouring pairs of meaningful word stems, such as ("not", "bite")."""
    kept = NEGATIONS | ORDER_WORDS
    stems = [_stem(word) for word in words if word not in FILLER_WORDS or word in kept]
    return set(pairwise(stems))


def _stem(word: str) -> str:
    """Return a rough stem, so "wears" and "wear", or "eating" and "eat", compare equal."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _words(text: str) -> list[str]:
    """Return lowercase words, with "n't" turned into "not", so negations compare equal."""
    text = re.sub(r"n't\b", " not", text.lower().replace("’", "'"))
    return re.findall(r"[a-z0-9]+", text)


def fix_phone_numbers(text: str, numbers: list[str]) -> tuple[str, bool]:
    """Rewrite any number close to one of these as that number exactly. Return the text and
    whether anything changed."""
    exact = {number: re.sub(r"\D", "", number) for number in numbers}

    def replace(match: re.Match[str]) -> str:
        written = match.group()
        digits = re.sub(r"\D", "", written)
        if len(digits) < MIN_PHONE_DIGITS:
            return written
        # The closest number wins. A tie goes to the first, the general clinic line.
        number = min(numbers, key=lambda item: _edits(digits, exact[item]))
        return number if _edits(digits, exact[number]) <= MAX_PHONE_EDITS else written

    fixed = DIGIT_RUN.sub(replace, text)
    return fixed, fixed != text


def _edits(a: str, b: str) -> int:
    """Return how many single-character edits turn a into b."""
    # row[j] is the edits from the part of a read so far to the first j characters of b.
    row = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        diagonal, row[0] = row[0], i
        for j, char_b in enumerate(b, 1):
            change = diagonal + (char_a != char_b)
            diagonal = row[j]
            row[j] = min(row[j] + 1, row[j - 1] + 1, change)
    return row[-1]


def claimed_kinds(text: str, claim_phrases: dict[str, list[str]]) -> list[str]:
    """Return each kind of change the text claims, in any letter case.

    The text is split into clauses at punctuation followed by a space, so "10.30" stays whole.
    A phrase counts unless an offer opens its clause and leads straight to it. Missing a claim
    is the unsafe mistake, so anything else counts, even inside a question.
    """
    clauses = [clause.strip() for clause in CLAUSE_BREAK.split(text.lower())]
    return [
        kind
        for kind, phrases in claim_phrases.items()
        if any(_claims(clause, phrase) for clause in clauses for phrase in phrases)
    ]


def _claims(clause: str, phrase: str) -> bool:
    """Return True if the phrase appears in the clause other than as the object of an offer."""
    start = clause.find(phrase)
    while start != -1:
        before = clause[:start]
        offered = any(
            before.startswith(opener) and len(before[len(opener) :].split()) <= OFFER_GAP
            for opener in OFFER_OPENERS
        )
        if not offered:
            return True
        start = clause.find(phrase, start + 1)
    return False


def _has_word(text: str, words: list[str]) -> bool:
    """Return True if any of these words appears as a whole word."""
    return any(re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) for word in words)


def _trim(text: str, max_sentences: int) -> str:
    """Keep the first sentences, cutting only at a sentence end."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:max_sentences])


def check_lead_in(
    text: str,
    service_id: str,
    actions: list[ActionRecord],
    found_appointments: list[Appointment],
    clinic: Clinic,
    model_claim: bool = False,
    model_advice: bool = False,
    sources: Sequence[str] = (),
) -> ReplyDecision:
    """Check the sentence the model sent with a prepare call. A failed one becomes empty."""
    if not text.strip():
        return ReplyDecision(text="", replaced=False)

    # Step 1: the normal reply check, cut to one sentence. A replaced reply is dropped whole.
    base = check_reply(
        text, actions, found_appointments, clinic, 1, model_claim, model_advice, sources
    )
    if base.replaced:
        return ReplyDecision(text="", replaced=True, reason=base.reason)

    # Step 2: no question, so the caller's yes can only answer the code's question.
    if "?" in text:
        return ReplyDecision(text="", replaced=True, reason="question")

    # Step 3: no day, time or date, so the sentence cannot disagree with the question.
    lowered = text.lower()
    if _has_word(lowered, list(WHEN_WORDS)) or any(
        re.search(pattern, lowered) for pattern in WHEN_PATTERNS
    ):
        return ReplyDecision(text="", replaced=True, reason="time")

    # Step 4: no other service, for the same reason. The booked one may be named.
    plain = _plain(lowered)
    for service in clinic.services:
        if service.id != service_id and any(
            re.search(rf"\b{re.escape(name)}\b", plain) for name in _service_names(service)
        ):
            return ReplyDecision(text="", replaced=True, reason="service")

    return ReplyDecision(text=base.text, replaced=False, reason=base.reason)


def _plain(text: str) -> str:
    """Lowercase text with hyphens and underscores as spaces, so spellings compare equal."""
    return re.sub(r"[-_]", " ", text.lower())


def _service_names(service: Service) -> set[str]:
    """Return the ways a caller or model might write a service, such as check-up or checkup."""
    names = {_plain(service.name), _plain(service.id)}
    # The first two words of a longer name, such as "first visit" from first-visit consultation.
    names |= {" ".join(name.split()[:2]) for name in names if len(name.split()) > 2}
    names |= {name.replace(" ", "") for name in names}
    return names
