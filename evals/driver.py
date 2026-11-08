"""A Speech object that plays the caller. Stage: 4.

What this file is:
    Feeds scripted caller lines into the normal session loop.

What it will contain:
    - ScenarioSpeech.

How to understand it:
    It looks like a speech backend to the session, so scenarios run the same code as a real
    call. It answers "yes" on the turn after a confirmation question when auto_yes is true,
    because the model's wording can change between runs and a fixed script would drift out of
    step.

Uses:
    speech/base.py

Used by:
    evals/evaluate.py
"""
