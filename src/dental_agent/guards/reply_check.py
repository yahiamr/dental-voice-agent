"""Checks the model's text before it is spoken. Stage: 1.

What this file is:
    The last check on anything the model wrote.

What it contains:
    - check_reply(text, actions, found_appointments, clinic, max_sentences, model_claim,
      model_advice) -> ReplyDecision.
    - check_lead_in(text, service_id, actions, found_appointments, clinic, model_claim,
      model_advice) -> ReplyDecision.
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
    the spoken text of a real action of that kind or a found appointment. If it does not, the
    reply becomes status_text. That is why spoken strings must be exact.

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
    2. Claim phrases, kind by kind, each needing proof of the same kind. The reason is claim.
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

from dental_agent.clinic.data import Clinic, Service
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
) -> ReplyDecision:
    """Check the model's reply, and return the text that is safe to speak.

    model_claim and model_advice are the checker's answers. Both false gives the result on main.
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
    if advice:
        reason = "dosing" if dosing_hit else "model_advice"
        return ReplyDecision(text=clinic.line("medical"), replaced=True, reason=reason)

    # Step 4: any claim of a change must be backed by a real record.
    results = [item.result_text for item in actions]
    found_spoken = [item.spoken.lower() for item in found_appointments if item.spoken]
    for kind in matched:
        # The reply must contain the exact time of a same-kind action or a found appointment.
        proof = [item.spoken.lower() for item in actions if item.kind == kind] + found_spoken
        if not any(spoken in lowered for spoken in proof):
            return ReplyDecision(text=status_text(results), replaced=True, reason="claim")

    # Step 4b: a claim only the checker saw has no kind, so a record of any kind is the proof.
    if model_claim and not matched:
        proof = [item.spoken.lower() for item in actions if item.spoken] + found_spoken
        if not any(spoken in lowered for spoken in proof):
            return ReplyDecision(text=status_text(results), replaced=True, reason="model_claim")

    # Step 5: trim, because long replies are hard to follow by ear.
    reason = "phone" if fixed else None
    return ReplyDecision(text=_trim(cleaned, max_sentences), replaced=False, reason=reason)


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
) -> ReplyDecision:
    """Check the sentence the model sent with a prepare call. A failed one becomes empty."""
    if not text.strip():
        return ReplyDecision(text="", replaced=False)

    # Step 1: the normal reply check, cut to one sentence. A replaced reply is dropped whole.
    base = check_reply(text, actions, found_appointments, clinic, 1, model_claim, model_advice)
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
