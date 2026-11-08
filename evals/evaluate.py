"""The evaluation runner. Stage: 3.

What this file is:
    Runs the scenarios and writes the results. Run it with uv run python -m evals.evaluate.

What it will contain:
    - Runs each scenario 3 times with a fresh database.
    - --provider and --model: run on another provider.
    - --voice-traces <dir>: adds delay figures.
    - --retrieval: runs the labelled questions, with no key needed.

How to understand it:
    A scenario passes when at least 2 of its 3 runs pass, because a real model can vary. Results
    go to runs/results.md. Guard tests in tests/ have a stricter rule: they must pass every
    time.

Uses:
    factories.py, session.py, evals/driver.py, evals/metrics.py

Used by:
    You
"""
