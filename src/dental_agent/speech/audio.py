"""Microphone and speaker helpers. Stage: 2.

What this file is:
    Low-level audio, using sounddevice.

What it contains:
    - MIC_RATE (16 kHz, used when no device rate is known), FRAME_MS (50 ms) and PLAY_RATE
      (24 kHz).
    - make_tone(rate, hz, ms): a short beep as 16-bit mono audio.
    - Microphone: open(), start_listening(), stop_listening(), read(timeout), clear(), close().
    - Speaker: play(pcm, on_first_audio), play_stream(chunks, on_first_audio), latency and
      close().

How to understand it:
    The microphone records mono at the device's own rate, in blocks of the device's choosing,
    and cuts them into 50 ms frames. Forcing 16 kHz in fixed blocks gave the MacBook microphone
    0.66 s of input delay, against 0.12 s at its own 48 kHz. Deepgram is told the rate. The
    microphone stays open for the whole call, but a frame is kept only while the caller's turn
    is open. Sound captured while the agent is
    thinking or speaking is thrown away, so the bot cannot hear itself. Starting a turn clears
    anything waiting, so each turn starts clean.

    The speaker plays 16-bit mono audio as it arrives, on one output stream kept open for the
    call. It holds the first 200 ms before starting, so a slow first chunk does not leave a gap
    in the middle of a word. A chunk can end in the middle of a sample, so an odd last byte is
    carried to the next chunk.

    Writing returns once audio is queued, not heard, and the device holds up to its latency.
    Bluetooth earbuds reported a whole second. Stopping the stream after each line dropped that
    queued tail, which cut the last half second of speech and the tone. So the stream is never
    stopped between lines, and play waits until the line's length plus the device's latency has
    passed. The mic reopens only after that, so it never hears the bot. A stream that fails is
    closed and replaced on the next line.

    sounddevice is imported only when a stream is opened, so this file loads, and its tests
    run, without the voice extra. Tests pass a fake stream_factory.

Uses:
    sounddevice, only when a stream is opened

Used by:
    speech/deepgram.py, speech/local.py
"""

import math
import queue
import struct
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

MIC_RATE = 16000  # Deepgram is sent 16 kHz mono linear16.
FRAME_MS = 50  # One microphone frame.
PLAY_RATE = 24000  # Deepgram speech is asked for at this rate.
PREBUFFER_MS = 200  # Audio held before playback starts.
FADE_MS = 10  # Fade at each end of the tone, so it does not click.


def make_tone(rate: int = PLAY_RATE, hz: int = 880, ms: int = 150, volume: float = 0.3) -> bytes:
    """Return a short beep as 16-bit mono audio, faded at both ends."""
    count = rate * ms // 1000
    fade = max(1, rate * FADE_MS // 1000)
    samples = []
    for index in range(count):
        # The fade rises from 0 at the first sample and falls to 0 at the last.
        envelope = min(1.0, index / fade, (count - 1 - index) / fade)
        value = volume * envelope * math.sin(2 * math.pi * hz * index / rate)
        samples.append(int(value * 32767))
    return struct.pack(f"<{count}h", *samples)


def _input_stream(**options: Any) -> Any:
    """Open a real microphone stream. Imported here, because it needs the voice extra."""
    import sounddevice

    return sounddevice.RawInputStream(**options)


def _output_stream(**options: Any) -> Any:
    """Open a real speaker stream. Imported here, because it needs the voice extra."""
    import sounddevice

    return sounddevice.RawOutputStream(**options)


def _default_input_rate() -> float:
    """Return the default microphone's own sample rate. Imported here, as it needs the extra."""
    import sounddevice

    return sounddevice.query_devices(kind="input")["default_samplerate"]


class Microphone:
    """The caller's microphone, kept open for the call, keeping frames only while listening."""

    def __init__(
        self,
        rate: int | None = None,
        stream_factory: Callable[..., Any] = _input_stream,
        rate_lookup: Callable[[], float] = _default_input_rate,
    ):
        self.rate = rate  # None means the device's own rate, read when the stream opens.
        self._factory = stream_factory
        self._rate_lookup = rate_lookup
        self._stream: Any = None
        self._frames: queue.Queue[bytes] = queue.Queue()
        self._pending = b""  # Audio from the device not yet a whole frame.
        self._lock = threading.Lock()
        self._listening = False

    def open(self) -> None:
        """Start recording. Calling it again does nothing."""
        if self._stream is not None:
            return
        if self.rate is None:
            self.rate = int(self._rate_lookup())
        # Blocks of the device's choosing, at its own rate, keep its input delay low.
        self._stream = self._factory(
            samplerate=self.rate,
            channels=1,
            dtype="int16",
            blocksize=0,
            latency="low",
            callback=self._on_audio,
        )
        self._stream.start()

    @property
    def frame_bytes(self) -> int:
        """Return the size of one 50 ms frame in bytes."""
        return (self.rate or MIC_RATE) * FRAME_MS // 1000 * 2

    def _on_audio(self, data: Any, frames: int, time_info: Any, status: Any) -> None:
        """Called by sounddevice on its own thread with each block, cut here into frames."""
        with self._lock:
            if not self._listening:
                return
            self._pending += bytes(data)
            size = self.frame_bytes
            while len(self._pending) >= size:
                self._frames.put(self._pending[:size])
                self._pending = self._pending[size:]

    def start_listening(self) -> None:
        """Open the caller's turn, dropping anything captured before it."""
        with self._lock:
            self._drop_waiting()
            self._listening = True

    def stop_listening(self) -> None:
        """Close the caller's turn, so later sound is thrown away."""
        with self._lock:
            self._listening = False
            self._drop_waiting()

    def read(self, timeout: float) -> bytes | None:
        """Return the next frame, or None if none arrives in time."""
        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear(self) -> None:
        """Drop every frame waiting to be read."""
        with self._lock:
            self._drop_waiting()

    def _drop_waiting(self) -> None:
        """Drop waiting frames and any part frame. The caller holds the lock."""
        self._pending = b""
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        """Stop recording."""
        self._listening = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class Speaker:
    """Plays 16-bit mono audio as it arrives, on one stream kept open for the call."""

    def __init__(
        self,
        rate: int = PLAY_RATE,
        prebuffer_ms: int = PREBUFFER_MS,
        stream_factory: Callable[..., Any] = _output_stream,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.rate = rate
        self._prebuffer = rate * prebuffer_ms // 1000 * 2  # In bytes.
        self._factory = stream_factory
        self._clock = clock
        self._sleep = sleep
        self._stream: Any = None

    @property
    def latency(self) -> float:
        """Return the output device's delay in seconds, or 0 before the stream opens."""
        return float(getattr(self._stream, "latency", 0.0) or 0.0)

    def play(self, pcm: bytes, on_first_audio: Callable[[], None] = lambda: None) -> None:
        """Play one piece of audio and wait until it has been heard."""
        self.play_stream([pcm], on_first_audio)

    def play_stream(self, chunks: Iterable[bytes], on_first_audio: Callable[[], None]) -> None:
        """Play chunks as they arrive, and wait until the last one has been heard.

        on_first_audio is called once, just before the first audio is written.
        """
        held = b""
        started_at: float | None = None
        samples = 0
        try:
            for chunk in chunks:
                held += chunk
                if started_at is None and len(held) < self._prebuffer:
                    continue
                # Keep whole samples only. An odd byte waits for the next chunk.
                whole = len(held) - len(held) % 2
                if whole == 0:
                    continue
                started_at = self._begin(started_at, on_first_audio)
                self._stream.write(held[:whole])
                samples += whole // 2
                held = held[whole:]
            # Short audio may never fill the buffer, and it must still play.
            if held:
                started_at = self._begin(started_at, on_first_audio)
                self._stream.write(held + b"\x00" * (len(held) % 2))
                samples += (len(held) + 1) // 2
        except BaseException:
            self.close()  # A broken stream is replaced on the next line.
            raise
        if started_at is not None:
            # Writing returns once audio is queued, not heard. The device still holds up to its
            # latency, which was a whole second on Bluetooth earbuds, so wait for that too.
            heard_at = started_at + samples / self.rate + self.latency
            remaining = heard_at - self._clock()
            if remaining > 0:
                self._sleep(remaining)

    def _begin(self, started_at: float | None, on_first_audio: Callable[[], None]) -> float:
        """Open the stream if needed. Return when this line's first audio was written."""
        if started_at is not None:
            return started_at
        if self._stream is None:
            self._stream = self._factory(samplerate=self.rate, channels=1, dtype="int16")
            self._stream.start()
        on_first_audio()
        return self._clock()

    def close(self) -> None:
        """Stop and release the stream."""
        if self._stream is not None:
            stream, self._stream = self._stream, None
            try:
                stream.stop()
            finally:
                stream.close()
