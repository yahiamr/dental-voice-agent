"""Tests for the microphone and speaker helpers. Stage: 2.

What this file is:
    Tests speech/audio.py with fake sound streams, so no microphone or speaker is needed.

What it contains:
    - The tone: its length, sample format and quiet ends.
    - Microphone: frames are kept only while listening, and clearing drops what is waiting.
    - Speaker: audio is held until a short buffer fills, odd bytes are carried over, the first
      audio is reported once, and the stream is always closed.

Used by:
    pytest
"""

import struct

import pytest

from dental_agent.speech.audio import FRAME_MS, MIC_RATE, Microphone, Speaker, make_tone


class FakeInput:
    """Stands in for sounddevice.RawInputStream."""

    def __init__(self, **options):
        self.options = options
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        pass

    def close(self):
        self.closed = True


class FakeOutput:
    """Stands in for sounddevice.RawOutputStream, recording every write."""

    def __init__(self, fail_on_write=False, latency=0.0, **options):
        self.options = options
        self.latency = latency
        self.writes: list[bytes] = []
        self.started = False
        self.stopped = False
        self.closed = False
        self.fail_on_write = fail_on_write

    def start(self):
        self.started = True

    def write(self, data):
        if self.fail_on_write:
            raise RuntimeError("device gone")
        self.writes.append(bytes(data))

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def test_the_tone_is_16_bit_mono_with_quiet_ends():
    tone = make_tone(rate=24000, ms=150)
    assert len(tone) == 24000 * 150 // 1000 * 2  # Two bytes per sample.
    samples = struct.unpack(f"<{len(tone) // 2}h", tone)
    # A fade at each end avoids a click.
    assert abs(samples[0]) < 100 and abs(samples[-1]) < 100
    assert max(abs(value) for value in samples) > 5000


FRAME = b"\x01\x00" * (MIC_RATE * FRAME_MS // 1000)  # One 50 ms frame at 16 kHz.


def test_the_microphone_opens_once_at_the_devices_own_rate():
    made = []

    def factory(**options):
        made.append(FakeInput(**options))
        return made[-1]

    mic = Microphone(stream_factory=factory, rate_lookup=lambda: 48000.0)
    mic.open()
    mic.open()
    assert len(made) == 1 and made[0].started
    # The device's own rate, and blocks of its own choosing: on the MacBook microphone, 16 kHz
    # in fixed blocks reported 0.66 s of input delay, and 48 kHz in free blocks 0.12 s.
    assert made[0].options["samplerate"] == 48000 and mic.rate == 48000
    assert made[0].options["blocksize"] == 0
    assert made[0].options["latency"] == "low"
    assert made[0].options["channels"] == 1
    assert made[0].options["dtype"] == "int16"
    mic.close()
    assert made[0].closed


def test_a_set_rate_is_used_without_asking_the_device():
    def no_lookup():
        raise AssertionError("the device should not be asked")

    mic = Microphone(rate=MIC_RATE, stream_factory=FakeInput, rate_lookup=no_lookup)
    mic.open()
    assert mic.rate == MIC_RATE


def test_blocks_of_any_size_come_out_as_50_ms_frames():
    mic = Microphone(rate=MIC_RATE, stream_factory=FakeInput)
    mic.open()
    mic.start_listening()
    mic._on_audio(FRAME[:500], 250, None, None)
    assert mic.read(timeout=0.01) is None  # Not a whole frame yet.
    mic._on_audio(FRAME[500:] + FRAME[:100], 0, None, None)
    assert mic.read(timeout=0.01) == FRAME
    assert mic.read(timeout=0.01) is None


def test_frames_are_dropped_unless_listening():
    mic = Microphone(rate=MIC_RATE, stream_factory=FakeInput)
    mic.open()
    mic._on_audio(FRAME, 800, None, None)  # The bot is speaking: thrown away.
    mic.start_listening()
    mic._on_audio(FRAME, 800, None, None)
    assert mic.read(timeout=0.01) == FRAME
    assert mic.read(timeout=0.01) is None
    mic.stop_listening()
    mic._on_audio(FRAME, 800, None, None)
    assert mic.read(timeout=0.01) is None


def test_starting_to_listen_drops_frames_already_waiting():
    mic = Microphone(rate=MIC_RATE, stream_factory=FakeInput)
    mic.open()
    mic.start_listening()
    mic._on_audio(FRAME, 800, None, None)
    mic._on_audio(FRAME[:300], 150, None, None)  # A part frame is dropped too.
    mic.start_listening()  # A new turn starts clean.
    mic._on_audio(FRAME[:1300], 650, None, None)
    assert mic.read(timeout=0.01) is None


def test_the_speaker_holds_audio_until_its_buffer_fills():
    streams = []

    def factory(**options):
        streams.append(FakeOutput(**options))
        return streams[-1]

    speaker = Speaker(rate=1000, prebuffer_ms=100, stream_factory=factory)
    first = []
    # 100 ms at 1 kHz is 100 samples, or 200 bytes. The first two chunks are 120 bytes.
    speaker.play_stream([b"\x00" * 60, b"\x00" * 60, b"\x00" * 200], lambda: first.append(1))
    stream = streams[0]
    assert stream.options["samplerate"] == 1000
    assert stream.writes[0] == b"\x00" * 320  # Held, then written together.
    assert first == [1]
    # The stream stays open for the next line. Stopping it would drop audio still queued.
    assert not stream.stopped and not stream.closed


def test_the_speaker_carries_an_odd_byte_to_the_next_chunk():
    streams = []

    def factory(**options):
        streams.append(FakeOutput(**options))
        return streams[-1]

    speaker = Speaker(rate=1000, prebuffer_ms=0, stream_factory=factory)
    speaker.play_stream([b"\x01\x02\x03", b"\x04"], lambda: None)
    assert b"".join(streams[0].writes) == b"\x01\x02\x03\x04"
    assert all(len(item) % 2 == 0 for item in streams[0].writes)


def test_the_speaker_plays_short_audio_that_never_fills_the_buffer():
    streams = []

    def factory(**options):
        streams.append(FakeOutput(**options))
        return streams[-1]

    first = []
    Speaker(rate=1000, prebuffer_ms=500, stream_factory=factory).play(
        b"\x00" * 10, lambda: first.append(1)
    )
    assert streams[0].writes == [b"\x00" * 10]
    assert first == [1]


def test_the_speaker_closes_its_stream_when_playback_fails():
    streams = []

    def factory(**options):
        streams.append(FakeOutput(fail_on_write=True, **options))
        return streams[-1]

    with pytest.raises(RuntimeError):
        Speaker(rate=1000, prebuffer_ms=0, stream_factory=factory).play(b"\x00" * 4)
    assert streams[0].closed


def test_the_speaker_opens_no_stream_for_no_audio():
    streams = []

    def factory(**options):
        streams.append(FakeOutput(**options))
        return streams[-1]

    first = []
    Speaker(stream_factory=factory).play_stream([], lambda: first.append(1))
    assert streams == [] and first == []


class FakeTime:
    """A clock that moves only when sleep is called."""

    def __init__(self):
        self.now = 100.0
        self.slept: list[float] = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def test_one_speaker_stream_serves_the_whole_call():
    streams = []

    def factory(**options):
        streams.append(FakeOutput(**options))
        return streams[-1]

    speaker = Speaker(rate=1000, prebuffer_ms=0, stream_factory=factory)
    speaker.play(b"\x00" * 20)
    speaker.play(b"\x00" * 20)
    assert len(streams) == 1 and streams[0].started
    speaker.close()
    assert streams[0].stopped and streams[0].closed


def test_playing_waits_until_the_audio_and_the_device_delay_have_passed():
    # Bluetooth earbuds reported a whole second of output delay. Returning earlier would let
    # the mic reopen, and the next line start, while the caller still hears this one.
    fake = FakeTime()
    speaker = Speaker(
        rate=1000,
        prebuffer_ms=0,
        stream_factory=lambda **options: FakeOutput(latency=0.5, **options),
        clock=fake.clock,
        sleep=fake.sleep,
    )
    speaker.play(b"\x00" * 200)  # 100 samples at 1 kHz is 0.1 s.
    assert sum(fake.slept) == pytest.approx(0.6)
    assert speaker.latency == 0.5


def test_a_failed_stream_is_replaced_on_the_next_line():
    streams = []

    def factory(**options):
        streams.append(FakeOutput(fail_on_write=not streams, **options))
        return streams[-1]

    speaker = Speaker(rate=1000, prebuffer_ms=0, stream_factory=factory)
    with pytest.raises(RuntimeError):
        speaker.play(b"\x00" * 4)
    speaker.play(b"\x00" * 4)
    assert len(streams) == 2 and streams[0].closed
    assert streams[1].writes == [b"\x00" * 4]
