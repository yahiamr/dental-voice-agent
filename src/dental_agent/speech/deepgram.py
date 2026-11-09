"""The Deepgram voice backend. Stage: 2.

What this file is:
    The speech backend used for the submission.

What it contains:
    - STT_MODEL (nova-3) and TTS_MODEL (aura-2-thalia-en).
    - listen_options(settings, expect_yes, keyterms): the live socket's settings for one turn.
    - TurnCollector: turns Deepgram's messages for one turn into a SpeechResult, and keeps
      live_text, the words heard so far, for the display.
    - DeepgramSpeech: transcribe(expect_yes) and speak(text), with backend_name "deepgram". An
      optional CallDisplay (speech/display.py) shows the call in the terminal: listening with
      the live words, the caller's line, thinking, the bot's line, and speaking.

How to understand it:
    Listening opens one live WebSocket per turn, so the end-of-speech wait can differ by turn:
    eos_wait_ms (700 ms) for a normal reply, and eos_wait_yes_ms (1300 ms) when a yes is
    expected, so "yes... but Thursday" is not cut in half. utterance_end_ms is 1000, or 2000
    when a yes is expected. The tone plays before the mic opens, so it is never sent. Then a
    sender thread streams 50 ms microphone frames and a receiver thread queues Deepgram's
    messages, while the calling thread reads the queue and watches the clock.

    TurnCollector keeps only final segments with words. A turn ends on speech_final, or on
    UtteranceEnd as the backup for noise, and only once words were heard. The segments are
    joined, and the lowest segment confidence is reported, because one bad segment is enough
    to distrust a yes. speech_end_at is when the last word ended: the stream's start time plus
    that word's end. Deepgram pads a segment with silence, so the segment's own end would make
    the delay look shorter than it is. A segment with no word timings uses its end instead.
    Speech shorter than min_speech_ms, measured from the first word's start to the last
    word's end, counts as no speech.

    A turn has three timeouts:
    - No words within silence_timeout_s is no_speech.
    - No message from Deepgram for speech_timeout_s is an error, as a stalled socket.
    - A socket that fails, fails to open, or sends an Error message is an error. A socket that
      closes after words keeps them.
    The session handles each result, so a failure never ends the call by itself.

    nova-3's key terms boost words the caller is likely to say that are rare elsewhere, such as
    the clinic's name. Speaking asks for raw 24 kHz linear16 audio, so no decoder is needed,
    and plays it as it arrives. first_audio_at is when the first audio was written to the
    speaker, and audio_end_at is when the last sample had been heard, after the device's own
    delay. output_latency_s records that delay, so the trace shows delay as the caller hears
    it. After playback, it waits mic_reopen_ms before the next turn may open the mic.

    With a fixture directory, every turn is saved as turn-NNN.wav, mono at the microphone's rate, and
    turn-NNN.json, holding Deepgram's text, confidence, the kind of result, expect_yes and an
    empty expected result to fill in by hand. tests/test_speech_fixtures.py reads them.

    The SDK is imported only when a real client is built, so this file loads, and its tests
    run, without the voice extra. SDK calls were checked against deepgram-sdk 7.17.0.

Uses:
    speech/base.py, speech/audio.py, config.py, deepgram-sdk

Used by:
    factories.py
"""

import array
import json
import logging
import math
import queue
import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dental_agent.config import Settings
from dental_agent.speech.audio import MIC_RATE, PLAY_RATE, Microphone, Speaker, make_tone
from dental_agent.speech.base import SpeakResult, SpeechResult

logger = logging.getLogger(__name__)

STT_MODEL = "nova-3"
TTS_MODEL = "aura-2-thalia-en"
PROVIDER = "deepgram"
UTTERANCE_END_MS = 1000  # Deepgram's smallest value.
UTTERANCE_END_YES_MS = 2000  # A pause after "yes" must not end the turn early.
POLL_S = 0.05  # How often the calling thread checks the clock.
JOIN_S = 0.5  # How long to wait for the sender thread to stop.
# Local end of speech. Deepgram's own end-of-speech signal proved slow on real microphone audio
# (a median 2.5 s, and once 11 s), so the turn also ends when the room has been quiet for the
# wait, by asking Deepgram to finalize.
# A frame louder than this, or 4 times the room's floor, is speech. On the first recorded call,
# speech ran 400 to 3000 and breaths and clicks 100 to 250, so 300 ignores those.
MIN_LOUD_RMS = 300
FLOOR_FRAMES = 6  # The first 300 ms of a turn set the room's floor.
FLOOR_FACTOR = 4
FINALIZE_WAIT_S = 1.5  # How long to wait for Deepgram's reply before using the interim text.

_CLOSED = object()  # Put on the queue when Deepgram closes the socket.


def end_wait_s(settings: Settings, expect_yes: bool) -> float:
    """Return how long the room must be quiet before the turn ends, in seconds."""
    return (settings.eos_wait_yes_ms if expect_yes else settings.eos_wait_ms) / 1000


class LoudnessTracker:
    """Tracks how long the caller has been quiet, from the microphone frames sent."""

    def __init__(self, min_rms: float = MIN_LOUD_RMS, rate: int = MIC_RATE):
        self.min_rms = min_rms
        self.rate = rate
        self.threshold = min_rms
        self.audio_s = 0.0  # Seconds of audio seen.
        self.last_loud_s: float | None = None  # When the last loud frame ended.
        self._floor: list[float] = []

    def add(self, frame: bytes) -> None:
        """Take one frame of 16-bit mono audio."""
        level = _rms(frame)
        if len(self._floor) < FLOOR_FRAMES:
            self._floor.append(level)
            if len(self._floor) == FLOOR_FRAMES:
                # The second quietest frame, so one early word does not raise the floor.
                floor = sorted(self._floor)[1]
                self.threshold = max(self.min_rms, FLOOR_FACTOR * floor)
        self.audio_s += len(frame) / 2 / self.rate
        if level > self.threshold:
            self.last_loud_s = self.audio_s

    @property
    def quiet_s(self) -> float | None:
        """Return seconds since the last loud frame, or None if nothing was loud yet."""
        if self.last_loud_s is None:
            return None
        return self.audio_s - self.last_loud_s


def listen_options(
    settings: Settings, expect_yes: bool, keyterms: list[str], sample_rate: int = MIC_RATE
) -> dict[str, Any]:
    """Return the live socket's settings for one turn, for audio at sample_rate."""
    options: dict[str, Any] = {
        "model": STT_MODEL,
        "language": "en",
        "encoding": "linear16",
        "sample_rate": sample_rate,
        "channels": 1,
        "interim_results": "true",
        "smart_format": "true",
        "endpointing": settings.eos_wait_yes_ms if expect_yes else settings.eos_wait_ms,
        "utterance_end_ms": UTTERANCE_END_YES_MS if expect_yes else UTTERANCE_END_MS,
    }
    if keyterms:
        options["keyterm"] = list(keyterms)  # The SDK repeats the parameter for each term.
    return options


class TurnCollector:
    """Turns Deepgram's messages for one turn into a SpeechResult."""

    def __init__(self, min_speech_ms: int):
        self.min_speech_ms = min_speech_ms
        self.heard_words = False
        # Each segment is (text, confidence, first word start, last word end), in stream seconds.
        self._segments: list[tuple[str, float, float, float]] = []
        # The latest result with words, final or not. Used only if Deepgram sends no final.
        self._interim: tuple[str, float, float, float] | None = None
        self._pending = ""  # Words heard after the last final segment, for the display.

    @property
    def live_text(self) -> str:
        """Return the words heard so far: the final segments, then any words not yet final."""
        parts = [segment[0] for segment in self._segments]
        if self._pending:
            parts.append(self._pending)
        return " ".join(parts)

    @property
    def has_segments(self) -> bool:
        """Return True once a final segment with words has arrived."""
        return bool(self._segments)

    def add(self, message: Any) -> bool:
        """Take one message. Return True when the turn has ended."""
        kind = getattr(message, "type", "")
        if kind == "UtteranceEnd":
            return self.has_segments
        if kind != "Results":
            return False
        alternative = message.channel.alternatives[0]
        text = (alternative.transcript or "").strip()
        if text:
            self.heard_words = True
            words = alternative.words or []
            first = words[0].start if words else message.start
            last = words[-1].end if words else message.start + message.duration
            segment = (text, alternative.confidence, first, last)
            self._interim = segment
            if message.is_final:
                self._segments.append(segment)
                self._pending = ""
            else:
                self._pending = text
        if getattr(message, "from_finalize", False):
            # Deepgram's reply to Finalize: everything it had is now final.
            return self.heard_words
        return bool(message.speech_final and self._segments)

    def result(self, stream_start: float, now: float, use_interim: bool = False) -> SpeechResult:
        """Return what was heard. stream_start is the wall time of the first audio sent.

        With use_interim, the latest interim text stands in when no segment is final.
        """
        segments = self._segments
        if not segments and use_interim and self._interim is not None:
            segments = [self._interim]
        if not segments:
            return SpeechResult(kind="no_speech", provider=PROVIDER)
        first_start = segments[0][2]
        last_end = segments[-1][3]
        if (last_end - first_start) * 1000 < self.min_speech_ms:
            return SpeechResult(kind="no_speech", provider=PROVIDER)
        return SpeechResult(
            kind="speech",
            text=" ".join(segment[0] for segment in segments),
            confidence=min(segment[1] for segment in segments),
            timings={"speech_end_at": stream_start + last_end, "transcript_at": now},
            provider=PROVIDER,
        )


class DeepgramSpeech:
    """Listens and speaks through Deepgram, with the Mac's microphone and speaker."""

    uses_tone = True  # The greeting tells the caller to speak after the tone.
    backend_name = PROVIDER  # Written to the trace.

    def __init__(
        self,
        settings: Settings,
        api_key: str,
        keyterms: list[str],
        fixture_dir: Path | str | None = None,
        display: Any = None,
        client: Any = None,
        mic: Any = None,
        speaker: Any = None,
        now: Callable[[], float] = time.time,
    ):
        self.settings = settings
        self.keyterms = keyterms
        self.fixture_dir = Path(fixture_dir) if fixture_dir else None
        # A CallDisplay shows the call in the terminal. None shows nothing, as in the tests.
        self.display = display
        self._client = client if client is not None else _make_client(api_key)
        self._mic = mic if mic is not None else Microphone()
        self._speaker = speaker if speaker is not None else Speaker()
        self._now = now  # Wall time, so timings from listening and speaking can be compared.
        self._tone = make_tone(rate=PLAY_RATE)
        self._turns_saved = 0

    def transcribe(self, expect_yes: bool) -> SpeechResult:
        """Listen for one reply. expect_yes waits longer for the caller to finish."""
        collector = TurnCollector(self.settings.min_speech_ms)
        recorded = bytearray()
        try:
            # Opened first, because Deepgram must be told the microphone's own rate.
            self._mic.open()
            rate = self._mic.rate
            options = listen_options(self.settings, expect_yes, self.keyterms, rate)
            with self._client.listen.v1.connect(**options) as connection:
                heard = self._listen(connection, collector, recorded, expect_yes)
        except Exception as error:  # A speech failure must not end the call by itself.
            logger.warning("Deepgram listening failed: %s", error)
            heard = SpeechResult(kind="error", provider=PROVIDER)
        finally:
            self._mic.stop_listening()
        if self.fixture_dir is not None:
            self._save_fixture(recorded, heard, expect_yes)
        if self.display is not None:
            if heard.kind == "speech":
                self.display.caller(heard.text)
            # The session runs the agent next, until it speaks or listens again.
            self.display.thinking()
        return heard

    def _listen(
        self, connection: Any, collector: TurnCollector, recorded: bytearray, expect_yes: bool
    ) -> SpeechResult:
        """Stream the mic to an open socket until the turn ends, and return what was heard."""
        self._speaker.play(self._tone)  # Before listening starts, so the tone is never sent.
        messages: queue.Queue[Any] = queue.Queue()
        stop = threading.Event()
        loudness = LoudnessTracker(rate=self._mic.rate)
        self._mic.start_listening()
        if self.display is not None:
            self.display.listening()
        stream_start = self._now()
        sender = threading.Thread(
            target=self._send, args=(connection, messages, stop, recorded, loudness), daemon=True
        )
        receiver = threading.Thread(target=_receive, args=(connection, messages), daemon=True)
        sender.start()
        receiver.start()
        try:
            outcome = self._wait(connection, collector, messages, loudness, expect_yes)
        finally:
            stop.set()
            sender.join(JOIN_S)
        self._mic.stop_listening()
        try:
            connection.send_close_stream()
        except Exception as error:  # The turn is over, so a failed goodbye changes nothing.
            logger.info("Could not close the Deepgram stream: %s", error)
        if outcome == "ended":
            return collector.result(stream_start, self._now(), use_interim=True)
        if outcome == "silence":
            return SpeechResult(kind="no_speech", provider=PROVIDER)
        return SpeechResult(kind="error", provider=PROVIDER)

    def _send(
        self,
        connection: Any,
        messages: queue.Queue[Any],
        stop: threading.Event,
        recorded: bytearray,
        loudness: LoudnessTracker,
    ) -> None:
        """Send microphone frames until told to stop. Runs on its own thread."""
        while not stop.is_set():
            frame = self._mic.read(timeout=POLL_S)
            if frame is None:
                continue
            recorded.extend(frame)
            loudness.add(frame)
            try:
                connection.send_media(frame)
            except Exception as error:  # Reported to the calling thread as a failed turn.
                messages.put(error)
                return

    def _wait(
        self,
        connection: Any,
        collector: TurnCollector,
        messages: queue.Queue[Any],
        loudness: LoudnessTracker,
        expect_yes: bool,
    ) -> str:
        """Read messages until the turn ends. Return ended, silence or error."""
        started = last_message = time.monotonic()
        wait_s = end_wait_s(self.settings, expect_yes)
        finalized_at: float | None = None
        while True:
            try:
                message = messages.get(timeout=POLL_S)
            except queue.Empty:
                message = None
            now = time.monotonic()
            if message is _CLOSED:
                # Deepgram closed the socket. Words heard before that still count.
                return "ended" if collector.has_segments else "error"
            if isinstance(message, Exception):
                logger.warning("Deepgram socket failed: %s", message)
                return "error"
            if message is not None:
                last_message = now
                if getattr(message, "type", "") == "Error":
                    logger.warning("Deepgram error: %s", getattr(message, "description", ""))
                    return "error"
                ended = collector.add(message)
                if self.display is not None and collector.heard_words:
                    self.display.words(collector.live_text)
                if ended:
                    return "ended"
            if not collector.heard_words and now - started > self.settings.silence_timeout_s:
                return "silence"
            quiet = loudness.quiet_s
            # Words were heard and the room has been quiet for the wait, so ask for the final.
            if finalized_at is None and collector.heard_words and (quiet or 0) >= wait_s:
                finalized_at = now
                try:
                    connection.send_finalize()
                except Exception as error:  # Deepgram's own end of speech still applies.
                    logger.info("Could not ask Deepgram to finalize: %s", error)
            if finalized_at is not None and now - finalized_at > FINALIZE_WAIT_S:
                return "ended"  # No reply to Finalize, so the interim text is used.
            if now - last_message > self.settings.speech_timeout_s:
                return "error"

    def speak(self, text: str, asks_confirmation: bool = False) -> SpeakResult:
        """Say one reply. asks_confirmation is for test drivers, and is ignored here."""
        timings: dict[str, float] = {}
        if self.display is not None:
            self.display.bot(text)
            self.display.speaking()
        try:
            chunks = self._client.speak.v1.audio.generate(
                text=text,
                model=TTS_MODEL,
                encoding="linear16",
                sample_rate=PLAY_RATE,
                container="none",
                # One retry, as the model call has: asking for speech twice has no side effect.
                request_options={"timeout": self.settings.speech_timeout_s, "max_retries": 1},
            )
            self._speaker.play_stream(
                chunks, lambda: timings.setdefault("first_audio_at", self._now())
            )
            timings["audio_end_at"] = self._now()
            # first_audio_at is when audio was handed to the device. The caller hears it this
            # much later, so delay as heard is first_audio_at plus this.
            timings["output_latency_s"] = float(getattr(self._speaker, "latency", 0.0))
        except Exception as error:  # The session prints a line that could not be spoken.
            logger.warning("Deepgram speech failed: %s", error)
            ok = False
        else:
            time.sleep(self.settings.mic_reopen_ms / 1000)
            ok = True
        if self.display is not None:
            # Either a filler line while the agent still works, or the reply before listening.
            self.display.thinking()
        return SpeakResult(ok=ok, timings=timings)

    def close(self) -> None:
        """Release the microphone and the speaker, and clear the display."""
        if self.display is not None:
            self.display.close()
        self._mic.close()
        close_speaker = getattr(self._speaker, "close", None)
        if close_speaker is not None:
            close_speaker()

    def _save_fixture(self, recorded: bytearray, heard: SpeechResult, expect_yes: bool) -> None:
        """Save the turn's audio and Deepgram's answer, for tests/test_speech_fixtures.py."""
        self._turns_saved += 1
        stem = self.fixture_dir / f"turn-{self._turns_saved:03d}"
        try:
            self.fixture_dir.mkdir(parents=True, exist_ok=True)
            with wave.open(str(stem.with_suffix(".wav")), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(self._mic.rate or MIC_RATE)
                audio.writeframes(bytes(recorded))
            record = {
                "text": heard.text,
                "confidence": heard.confidence,
                "kind": heard.kind,
                "expect_yes": expect_yes,
                "expected": None,  # Filled in by hand before the fixture is used in a test.
            }
            stem.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
        except OSError as error:  # A fixture that cannot be saved must not break the call.
            logger.warning("Could not save fixture %s: %s", stem, error)


def _receive(connection: Any, messages: queue.Queue[Any]) -> None:
    """Queue every message from the socket, then mark it closed. Runs on its own thread."""
    try:
        for message in connection:
            messages.put(message)
        messages.put(_CLOSED)
    except Exception as error:  # Reported to the calling thread as a failed turn.
        messages.put(error)


def _rms(frame: bytes) -> float:
    """Return the loudness of 16-bit mono audio, as root mean square."""
    samples = array.array("h", frame[: len(frame) - len(frame) % 2])
    if not samples:
        return 0.0
    return math.sqrt(sum(value * value for value in samples) / len(samples))


def _make_client(api_key: str) -> Any:
    """Build the real Deepgram client. Imported here, because it needs the voice extra."""
    from deepgram import DeepgramClient

    # One socket per turn, so a dropped socket is a failed turn, not a silent reconnect.
    return DeepgramClient(api_key=api_key, reconnect=False)
