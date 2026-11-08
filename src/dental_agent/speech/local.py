"""An on-device speech backend. Stage: Optional.

What this file is:
    An optional backend that runs on Apple Silicon with no outside service.

What it will contain:
    - LocalSpeech, using Silero, parakeet-mlx and Kokoro.

How to understand it:
    It replaces Deepgram only if it has lower word errors on the same recordings with delay
    within the targets.

Uses:
    speech/base.py, speech/audio.py

Used by:
    factories.py
"""
