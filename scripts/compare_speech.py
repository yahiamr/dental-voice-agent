"""Compares speech backends. Stage: Optional.

What this file is:
    Runs Deepgram and the local backend on the same recordings.

How to understand it:
    It reports word errors, end-of-speech delay and speech delay separately, using dates,
    Spanish names, yes and no. The result decides whether speech/local.py replaces Deepgram.

Uses:
    speech/, tests/fixtures/speech/

Used by:
    You
"""
