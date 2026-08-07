import numpy as np

from callmind.audio.vad import EnergyVAD, VADEventType

FRAME = 160


def silence(n_frames: int):
    for _ in range(n_frames):
        yield np.zeros(FRAME, dtype=np.int16)


def tone(n_frames: int, amp: int = 10000):
    phase = np.arange(n_frames * FRAME)
    samples = (np.sin(phase * 0.3) * amp).astype(np.int16)
    for i in range(n_frames):
        yield samples[i * FRAME : (i + 1) * FRAME]


def test_no_events_on_silence():
    vad = EnergyVAD()
    for frame in silence(100):
        assert vad.process_frame(frame) is None


def test_speech_start_then_end():
    vad = EnergyVAD(energy_dbfs=-45.0, start_frames=3, end_frames=5)
    events = []
    for frame in tone(20):
        ev = vad.process_frame(frame)
        if ev:
            events.append(ev.type)
    assert events == [VADEventType.SPEECH_START]
    assert vad.in_speech

    for frame in silence(20):
        ev = vad.process_frame(frame)
        if ev:
            events.append(ev.type)
    assert events == [VADEventType.SPEECH_START, VADEventType.SPEECH_END]
    assert not vad.in_speech


def test_short_blip_does_not_trigger():
    vad = EnergyVAD(energy_dbfs=-45.0, start_frames=3, end_frames=5)
    events = []
    for frame in tone(2):
        ev = vad.process_frame(frame)
        if ev:
            events.append(ev.type)
    for frame in silence(20):
        ev = vad.process_frame(frame)
        if ev:
            events.append(ev.type)
    assert events == []


def test_reset():
    vad = EnergyVAD(start_frames=3, end_frames=5)
    for frame in tone(10):
        vad.process_frame(frame)
    assert vad.in_speech
    vad.reset()
    assert not vad.in_speech
