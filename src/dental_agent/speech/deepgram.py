"""The Deepgram voice backend. Stage: 2.

What this file is:
    The speech backend used for the submission.

What it will contain:
    - DeepgramSpeech, which transcribes with nova-3 and speaks with aura-2-thalia-en.

How to understand it:
    It opens one live WebSocket per turn, so the end-of-speech wait can differ when a yes is
    expected. Final segments are joined, and the lowest segment confidence is reported, because
    one bad segment is enough to distrust a yes. The SDK calls get checked against current
    Deepgram docs at the start of Stage 2.

Uses:
    speech/base.py, speech/audio.py

Used by:
    factories.py
"""
