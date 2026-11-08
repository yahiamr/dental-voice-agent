"""The checker: a second model that answers yes-or-no questions code decides. Stage: checker.

What this file is:
    The package's public names, so other code imports from dental_agent.checker.

What it contains:
    Checker, CheckAnswer, CheckerError, CheckerChain, caller_questions and reply_questions.

How to understand it:
    The checker never writes to the calendar and never words what the caller hears. It answers
    closed questions, such as "is this an emergency?", and code decides what to do with the
    answer. Jev on OpenRouter answers first, then the Haiku checker. If both fail, the chain
    returns None and the fixed rules decide alone. See docs/designs/checker_model.md.

Used by:
    factories.py, deps.py, graph/nodes.py (from Step 3), evals/
"""

from dental_agent.checker.base import CheckAnswer, Checker, CheckerError
from dental_agent.checker.chain import CheckerChain
from dental_agent.checker.questions import caller_questions, reply_questions

__all__ = [
    "CheckAnswer",
    "Checker",
    "CheckerChain",
    "CheckerError",
    "caller_questions",
    "reply_questions",
]
