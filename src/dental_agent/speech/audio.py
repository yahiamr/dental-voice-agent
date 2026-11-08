"""Microphone and speaker helpers. Stage: 2.

What this file is:
    Low-level audio, using sounddevice.

What it will contain:
    - open_mic(), read_frames(), clear_input(), play(pcm, rate), play_tone().

How to understand it:
    Audio is 16 kHz, mono. Sound captured while the agent is thinking or speaking is thrown
    away, so the bot cannot hear itself. The input is cleared and a tone plays when it is the
    caller's turn.

Used by:
    speech/deepgram.py, speech/local.py
"""
