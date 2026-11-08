"""Tests for the Deepgram backend. Stage: 2.

What this file is:
    Tests speech/deepgram.py with a fake socket, microphone and speaker, so no key, network or
    sound device is needed.

What it contains:
    - TurnCollector: segment joining, the lowest confidence, which messages end a turn,
      speech_end_at from the last word, and too-short speech.
    - listen_options: the end-of-speech wait chosen by expect_yes, and the key terms.
    - transcribe: the tone before listening, silence, a stalled or failed socket, a Deepgram
      error message, and saved fixtures.
    - speak: the request sent, the timings, and a failed request.

Used by:
    pytest
"""

import json
import queue
import struct
import threading
import time
import wave
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from dental_agent.config import Settings
from dental_agent.speech import deepgram
from dental_agent.speech.deepgram import (
    STT_MODEL,
    TTS_MODEL,
    DeepgramSpeech,
    LoudnessTracker,
    TurnCollector,
    end_wait_s,
    listen_options,
)


def settings(**changes):
    """Fast settings, so timeouts in these tests take a fraction of a second."""
    base = {
        "silence_timeout_s": 0.3,
        "speech_timeout_s": 2.0,
        "mic_reopen_ms": 0,
        "min_speech_ms": 200,
    }
    return Settings(_env_file=None, **(base | changes))


def word(start, end):
    return SimpleNamespace(start=start, end=end)


def results(text, confidence=0.9, start=0.0, duration=1.0, final=True, speech_final=False,
            words=None, from_finalize=False):  # fmt: skip
    """A Deepgram Results message. By default one word spans most of the segment."""
    if words is None:
        words = [word(start + 0.1, start + duration - 0.1)] if text else []
    alternative = SimpleNamespace(transcript=text, confidence=confidence, words=words)
    return SimpleNamespace(
        type="Results",
        start=start,
        duration=duration,
        is_final=final,
        speech_final=speech_final,
        from_finalize=from_finalize,
        channel=SimpleNamespace(alternatives=[alternative]),
    )


UTTERANCE_END = SimpleNamespace(type="UtteranceEnd", last_word_end=1.0)


# ---------- TurnCollector ----------


def test_final_segments_are_joined_and_the_lowest_confidence_is_reported():
    collector = TurnCollector(min_speech_ms=200)
    assert not collector.add(results("Yes,", confidence=0.95, start=0.0, duration=0.8))
    assert collector.add(
        results("but Thursday.", confidence=0.6, start=0.8, duration=1.0, speech_final=True)
    )
    heard = collector.result(stream_start=100.0, now=103.0)
    assert heard.kind == "speech"
    assert heard.text == "Yes, but Thursday."
    assert heard.confidence == 0.6
    assert heard.provider == "deepgram"


def test_interim_results_are_not_kept():
    collector = TurnCollector(min_speech_ms=200)
    collector.add(results("Yes but", final=False))
    collector.add(results("Yes, but Monday.", speech_final=True))
    assert collector.result(stream_start=0.0, now=1.0).text == "Yes, but Monday."


def test_speech_end_comes_from_the_last_word_not_the_segment():
    collector = TurnCollector(min_speech_ms=200)
    # The segment runs to 3.0 s, padded with silence, but the last word ends at 1.4 s.
    collector.add(
        results("Yes.", start=0.0, duration=3.0, speech_final=True, words=[word(0.5, 1.4)])
    )
    heard = collector.result(stream_start=100.0, now=104.0)
    assert heard.timings["speech_end_at"] == pytest.approx(101.4)
    assert heard.timings["transcript_at"] == 104.0


def test_a_segment_without_words_uses_its_own_end():
    collector = TurnCollector(min_speech_ms=200)
    collector.add(results("Yes.", start=1.0, duration=0.5, speech_final=True, words=[]))
    assert collector.result(stream_start=10.0, now=12.0).timings["speech_end_at"] == 11.5


def test_utterance_end_ends_a_turn_only_after_words():
    collector = TurnCollector(min_speech_ms=200)
    assert not collector.add(UTTERANCE_END)
    collector.add(results("Tuesday."))
    assert collector.add(UTTERANCE_END)


def test_speech_final_with_no_words_does_not_end_the_turn():
    collector = TurnCollector(min_speech_ms=200)
    assert not collector.add(results("", speech_final=True))
    assert not collector.heard_words


def test_speech_shorter_than_the_minimum_counts_as_no_speech():
    collector = TurnCollector(min_speech_ms=200)
    collector.add(results("Uh.", speech_final=True, words=[word(0.50, 0.62)]))
    assert collector.result(stream_start=0.0, now=1.0).kind == "no_speech"


def test_a_finalize_reply_ends_the_turn_once_words_were_heard():
    collector = TurnCollector(min_speech_ms=200)
    assert not collector.add(results("", from_finalize=True))
    collector.add(results("Yes.", final=False))
    assert collector.add(results("", from_finalize=True))


def test_the_latest_interim_text_is_the_fallback():
    collector = TurnCollector(min_speech_ms=200)
    collector.add(results("Tues", final=False, confidence=0.5))
    collector.add(results("Tuesday morning", final=False, confidence=0.7))
    heard = collector.result(stream_start=0.0, now=1.0, use_interim=True)
    assert heard.text == "Tuesday morning" and heard.confidence == 0.7
    assert collector.result(stream_start=0.0, now=1.0).kind == "no_speech"


def test_final_segments_win_over_the_interim_fallback():
    collector = TurnCollector(min_speech_ms=200)
    collector.add(results("Yes.", final=True))
    collector.add(results("Yes. And", final=False))
    assert collector.result(stream_start=0.0, now=1.0, use_interim=True).text == "Yes."


def test_other_messages_are_ignored():
    collector = TurnCollector(min_speech_ms=200)
    assert not collector.add(SimpleNamespace(type="SpeechStarted", timestamp=0.3))
    assert not collector.add(SimpleNamespace(type="Metadata"))


# ---------- listen_options ----------


def test_a_normal_turn_ends_after_the_normal_wait():
    options = listen_options(settings(eos_wait_ms=700), expect_yes=False, keyterms=["Mirador"])
    assert options["endpointing"] == 700
    assert options["utterance_end_ms"] == 1000
    assert options["model"] == STT_MODEL == "nova-3"
    assert options["encoding"] == "linear16"
    assert options["sample_rate"] == 16000
    assert options["interim_results"] == "true"
    assert options["keyterm"] == ["Mirador"]


def test_a_turn_expecting_yes_waits_longer():
    options = listen_options(settings(eos_wait_yes_ms=1300), expect_yes=True, keyterms=[])
    assert options["endpointing"] == 1300
    assert options["utterance_end_ms"] == 2000
    assert "keyterm" not in options  # An empty list would send nothing useful.


# ---------- fakes for transcribe and speak ----------


class FakeConnection:
    """A live socket that yields scripted messages, then stays open until closed.

    after_finalize holds the messages Deepgram sends back once it is asked to finalize.
    """

    def __init__(self, messages, close_after=False, fail_with=None, after_finalize=()):
        self.messages = list(messages)
        self.close_after = close_after
        self.fail_with = fail_with
        self.after_finalize = list(after_finalize)
        self.later: queue.Queue = queue.Queue()
        self.sent: list[bytes] = []
        self.closed = threading.Event()
        self.close_stream_sent = False
        self.finalize_count = 0

    def __iter__(self):
        yield from self.messages
        if self.fail_with:
            raise self.fail_with
        if self.close_after:
            return
        while not self.closed.is_set():
            try:
                yield self.later.get(timeout=0.01)
            except queue.Empty:
                continue

    def send_media(self, data):
        self.sent.append(data)

    def send_finalize(self):
        self.finalize_count += 1
        for message in self.after_finalize:
            self.later.put(message)

    def send_close_stream(self):
        self.close_stream_sent = True


class FakeClient:
    """Stands in for DeepgramClient."""

    def __init__(self, connection=None, connect_error=None, chunks=(), speak_error=None):
        self.connection = connection
        self.connect_error = connect_error
        self.connect_options = None
        self.chunks = list(chunks)
        self.speak_error = speak_error
        self.speak_options = None
        self.listen = SimpleNamespace(v1=SimpleNamespace(connect=self._connect))
        audio = SimpleNamespace(generate=self._generate)
        self.speak = SimpleNamespace(v1=SimpleNamespace(audio=audio))

    @contextmanager
    def _connect(self, **options):
        self.connect_options = options
        if self.connect_error:
            raise self.connect_error
        try:
            yield self.connection
        finally:
            self.connection.closed.set()

    def _generate(self, **options):
        self.speak_options = options
        if self.speak_error:
            raise self.speak_error
        yield from self.chunks


SILENT = b"\x00\x00" * 800  # 50 ms at 16 kHz.


def frame(level):
    """A 50 ms frame whose loudness (RMS) is level."""
    return struct.pack("<h", level) * 800


class FakeMic:
    """Gives scripted 50 ms frames while listening, then repeats the last one.

    It records what happened. With no script, every frame is silent.
    """

    def __init__(self, events, frames=None, rate=16000):
        self.events = events
        self.rate = rate
        self.listening = False
        self.frames = list(frames or [SILENT])
        self.position = 0

    def open(self):
        self.events.append("open")

    def start_listening(self):
        self.listening = True
        self.events.append("listen")

    def stop_listening(self):
        self.listening = False
        self.events.append("stop")

    def read(self, timeout):
        time.sleep(0.005)
        if not self.listening:
            return None
        item = self.frames[min(self.position, len(self.frames) - 1)]
        self.position += 1
        return item

    def close(self):
        self.events.append("close")


class FakeSpeaker:
    """Records what was played and calls on_first_audio, as the real speaker does."""

    latency = 0.25  # Seconds of output delay, as a real device reports.

    def __init__(self, events, fail_with=None):
        self.events = events
        self.closed = False
        self.played: list[bytes] = []
        self.fail_with = fail_with

    def play(self, pcm, on_first_audio=lambda: None):
        self.events.append("tone")
        self.played.append(pcm)

    def play_stream(self, chunks, on_first_audio):
        on_first_audio()
        for chunk in chunks:
            if self.fail_with:
                raise self.fail_with
            self.played.append(chunk)
        self.events.append("played")

    def close(self):
        self.closed = True


def backend(client, events=None, fixture_dir=None, speaker_error=None, frames=None, **changes):
    events = [] if events is None else events
    return DeepgramSpeech(
        settings(**changes),
        api_key="test-key",
        keyterms=["Mirador"],
        fixture_dir=fixture_dir,
        client=client,
        mic=FakeMic(events, frames),
        speaker=FakeSpeaker(events, fail_with=speaker_error),
    )


# ---------- transcribe ----------


def test_a_reply_is_heard_after_the_tone():
    events = []
    connection = FakeConnection([results("I'd like a cleaning.", speech_final=True)])
    client = FakeClient(connection)
    heard = backend(client, events).transcribe(expect_yes=False)
    assert heard.kind == "speech"
    assert heard.text == "I'd like a cleaning."
    # The tone plays before the mic opens, so it is never sent to Deepgram.
    assert events.index("tone") < events.index("listen") < events.index("stop")
    assert client.connect_options["endpointing"] == 700
    assert connection.close_stream_sent


def test_expect_yes_reaches_the_socket():
    client = FakeClient(FakeConnection([results("Yes.", speech_final=True)]))
    backend(client).transcribe(expect_yes=True)
    assert client.connect_options["endpointing"] == 1300


def test_microphone_audio_is_sent_to_deepgram():
    connection = FakeConnection([])
    backend(FakeClient(connection), silence_timeout_s=0.1).transcribe(expect_yes=False)
    assert connection.sent and all(len(frame) == 1600 for frame in connection.sent)


def test_no_words_before_the_silence_timeout_is_no_speech():
    started = time.monotonic()
    heard = backend(FakeClient(FakeConnection([])), silence_timeout_s=0.2).transcribe(False)
    assert heard.kind == "no_speech"
    assert time.monotonic() - started < 1.5


def test_a_socket_that_stops_answering_is_an_error():
    # Words were heard, so the silence timeout no longer applies, then nothing more arrives.
    connection = FakeConnection([results("I'd like", final=False)])
    heard = backend(FakeClient(connection), speech_timeout_s=0.2).transcribe(False)
    assert heard.kind == "error"


def test_a_connection_that_fails_to_open_is_an_error():
    events = []
    heard = backend(FakeClient(connect_error=OSError("no network")), events).transcribe(False)
    assert heard.kind == "error"
    assert "listen" not in events


def test_a_deepgram_error_message_is_an_error():
    error = SimpleNamespace(type="Error", description="bad request")
    heard = backend(FakeClient(FakeConnection([error]))).transcribe(False)
    assert heard.kind == "error"


def test_a_socket_that_fails_mid_turn_is_an_error_and_stops_listening():
    events = []
    connection = FakeConnection([], fail_with=OSError("dropped"))
    heard = backend(FakeClient(connection), events).transcribe(False)
    assert heard.kind == "error"
    assert events[-1] == "stop"


def test_a_socket_that_closes_after_words_keeps_them():
    connection = FakeConnection([results("Thursday.")], close_after=True)
    heard = backend(FakeClient(connection)).transcribe(False)
    assert heard.kind == "speech" and heard.text == "Thursday."


def test_a_turn_is_saved_as_a_fixture(tmp_path):
    connection = FakeConnection([results("Yes please.", confidence=0.8, speech_final=True)])
    speech = backend(FakeClient(connection), fixture_dir=tmp_path)
    speech.transcribe(expect_yes=True)
    saved = json.loads((tmp_path / "turn-001.json").read_text())
    assert saved == {
        "text": "Yes please.",
        "confidence": 0.8,
        "kind": "speech",
        "expect_yes": True,
        "expected": None,
    }
    with wave.open(str(tmp_path / "turn-001.wav")) as audio:
        assert audio.getframerate() == 16000 and audio.getnchannels() == 1
        assert audio.getsampwidth() == 2 and audio.getnframes() > 0


# ---------- ending the turn on local silence ----------

QUIET, LOUD = frame(10), frame(3000)


def test_loudness_is_measured_against_the_rooms_own_floor():
    tracker = LoudnessTracker(min_rms=150)
    for _ in range(6):
        tracker.add(frame(100))  # A noisy room: the threshold becomes 400, not 150.
    tracker.add(frame(300))
    assert tracker.quiet_s is None  # Nothing loud yet.
    tracker.add(LOUD)
    tracker.add(frame(300))
    tracker.add(frame(300))
    assert tracker.quiet_s == pytest.approx(0.10)
    assert tracker.audio_s == pytest.approx(0.50)


def test_quiet_time_is_counted_at_the_mics_own_rate():
    tracker = LoudnessTracker(min_rms=150, rate=48000)
    tracker.add(struct.pack("<h", 3000) * 4800)  # 100 ms at 48 kHz.
    tracker.add(b"\x00\x00" * 2400)  # 50 ms of silence.
    assert tracker.audio_s == pytest.approx(0.15)
    assert tracker.quiet_s == pytest.approx(0.05)


def test_a_quiet_room_uses_the_minimum_threshold():
    tracker = LoudnessTracker(min_rms=150)
    for _ in range(6):
        tracker.add(QUIET)
    tracker.add(frame(200))
    assert tracker.quiet_s == 0.0


def test_the_wait_before_finalizing_is_longer_when_a_yes_is_expected():
    values = settings(eos_wait_ms=700, eos_wait_yes_ms=1300)
    assert end_wait_s(values, expect_yes=False) == 0.7
    assert end_wait_s(values, expect_yes=True) == 1.3


def test_finalize_after_local_silence_ends_the_turn_with_deepgrams_final_text():
    frames = [QUIET] * 6 + [LOUD] * 10 + [QUIET]
    connection = FakeConnection(
        [results("I'd like a", final=False)],
        after_finalize=[results("I'd like a cleaning.", from_finalize=True)],
    )
    started = time.monotonic()
    heard = backend(FakeClient(connection), frames=frames, eos_wait_ms=200).transcribe(False)
    assert heard.kind == "speech"
    assert heard.text == "I'd like a cleaning."
    assert connection.finalize_count == 1
    assert time.monotonic() - started < 2


def test_final_segments_before_finalize_are_kept():
    frames = [QUIET] * 6 + [LOUD] * 10 + [QUIET]
    connection = FakeConnection(
        [results("Yes.", final=True)],
        after_finalize=[results("I want a cleaning.", start=1.0, from_finalize=True)],
    )
    heard = backend(FakeClient(connection), frames=frames, eos_wait_ms=200).transcribe(False)
    assert heard.text == "Yes. I want a cleaning."


def test_an_empty_finalize_reply_still_ends_the_turn():
    frames = [QUIET] * 6 + [LOUD] * 10 + [QUIET]
    connection = FakeConnection(
        [results("Thursday.", final=True)],
        after_finalize=[results("", from_finalize=True)],
    )
    heard = backend(FakeClient(connection), frames=frames, eos_wait_ms=200).transcribe(False)
    assert heard.text == "Thursday."


def test_no_reply_to_finalize_falls_back_to_the_latest_interim_text(monkeypatch):
    monkeypatch.setattr(deepgram, "FINALIZE_WAIT_S", 0.2)
    frames = [QUIET] * 6 + [LOUD] * 10 + [QUIET]
    connection = FakeConnection([results("November 9 at ten", confidence=0.8, final=False)])
    heard = backend(FakeClient(connection), frames=frames, eos_wait_ms=200).transcribe(False)
    assert heard.kind == "speech"
    assert heard.text == "November 9 at ten"
    assert heard.confidence == 0.8


def test_no_finalize_while_the_caller_is_still_loud():
    connection = FakeConnection([results("I'd like", final=False)])
    speech = backend(
        FakeClient(connection),
        frames=[QUIET] * 6 + [LOUD],
        eos_wait_ms=100,
        speech_timeout_s=0.4,
    )
    speech.transcribe(False)
    assert connection.finalize_count == 0


def test_no_finalize_before_any_words_are_heard():
    connection = FakeConnection([])
    frames = [QUIET] * 6 + [LOUD] * 4 + [QUIET]
    backend(
        FakeClient(connection), frames=frames, eos_wait_ms=100, silence_timeout_s=0.5
    ).transcribe(False)
    assert connection.finalize_count == 0


def test_no_finalize_in_a_room_that_is_never_loud():
    # If the mic is too quiet to measure, Deepgram's own end of speech still ends the turn.
    connection = FakeConnection(
        [results("Hello", final=False), results("Hello.", speech_final=True)]
    )
    heard = backend(FakeClient(connection), frames=[QUIET], eos_wait_ms=100).transcribe(False)
    assert heard.text == "Hello."
    assert connection.finalize_count == 0


# ---------- speak ----------


def test_speak_streams_deepgram_audio_and_reports_timings():
    events = []
    client = FakeClient(chunks=[b"\x00\x00" * 10, b"\x00\x00" * 10])
    result = backend(client, events).speak("Your appointment is booked.")
    assert result.ok
    assert set(result.timings) == {"first_audio_at", "audio_end_at", "output_latency_s"}
    assert result.timings["output_latency_s"] == 0.25
    assert result.timings["first_audio_at"] <= result.timings["audio_end_at"]
    assert client.speak_options["text"] == "Your appointment is booked."
    assert client.speak_options["model"] == TTS_MODEL == "aura-2-thalia-en"
    assert client.speak_options["encoding"] == "linear16"
    assert client.speak_options["container"] == "none"
    assert client.speak_options["sample_rate"] == 24000
    assert client.speak_options["request_options"] == {"timeout": 2.0, "max_retries": 1}
    assert events == ["played"]


def test_a_failed_speech_request_is_not_ok():
    result = backend(FakeClient(speak_error=OSError("no network"))).speak("Hello.")
    assert not result.ok


def test_playback_that_fails_is_not_ok():
    result = backend(FakeClient(chunks=[b"\x00\x00"]), speaker_error=RuntimeError("gone")).speak(
        "Hello."
    )
    assert not result.ok


def test_the_backend_uses_a_tone_and_names_itself():
    speech = backend(FakeClient(FakeConnection([])))
    assert speech.uses_tone
    assert speech.backend_name == "deepgram"


def test_closing_releases_the_mic_and_the_speaker():
    events = []
    speech = backend(FakeClient(FakeConnection([])), events)
    speech.close()
    assert events[-1] == "close"
    assert speech._speaker.closed


def test_deepgram_is_told_the_mics_own_rate(tmp_path):
    events = []
    connection = FakeConnection([results("Yes.", speech_final=True)])
    client = FakeClient(connection)
    speech = DeepgramSpeech(
        settings(),
        api_key="test-key",
        keyterms=[],
        fixture_dir=tmp_path,
        client=client,
        mic=FakeMic(events, rate=48000),
        speaker=FakeSpeaker(events),
    )
    speech.transcribe(expect_yes=False)
    assert client.connect_options["sample_rate"] == 48000
    with wave.open(str(tmp_path / "turn-001.wav")) as audio:
        assert audio.getframerate() == 48000
