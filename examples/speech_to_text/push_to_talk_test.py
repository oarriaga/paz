from types import SimpleNamespace
from io import StringIO
from unittest.mock import patch

import numpy as np
import pytest

from examples.speech_to_text.microphone import build_sounddevice_error
from examples.speech_to_text.microphone import load_sounddevice
from examples.speech_to_text.microphone import verify_input_device
from examples.speech_to_text.push_to_talk import build_microphone_line
from examples.speech_to_text.push_to_talk import build_waveform
from examples.speech_to_text.push_to_talk import is_blank_text
from examples.speech_to_text.push_to_talk import list_microphones
from examples.speech_to_text.push_to_talk import PushToTalk


DEVICES = [
    {
        "name": "Built-in Audio",
        "max_input_channels": 2,
        "default_samplerate": 48000.0,
    },
    {
        "name": "HDMI Output",
        "max_input_channels": 0,
        "default_samplerate": 48000.0,
    },
    {
        "name": "USB Camera",
        "max_input_channels": 1,
        "default_samplerate": 44100.0,
    },
]


def test_build_waveform_handles_empty_recording():
    waveform = build_waveform([])
    assert waveform.dtype == np.float32
    assert waveform.shape == (0,)


def test_build_waveform_concatenates_chunks():
    chunks = [np.array([0.1, 0.2]), np.array([0.3], dtype="float64")]
    waveform = build_waveform(chunks)
    expected = np.array([0.1, 0.2, 0.3], dtype="float32")
    np.testing.assert_allclose(waveform, expected)


def test_load_sounddevice_raises_install_message():
    with patch("importlib.import_module", side_effect=ImportError):
        with pytest.raises(ImportError, match=build_sounddevice_error()):
            load_sounddevice()


def test_load_sounddevice_wraps_portaudio_error():
    with patch("importlib.import_module", side_effect=OSError):
        with pytest.raises(ImportError, match=build_sounddevice_error()):
            load_sounddevice()


def test_build_microphone_line_marks_selected_device():
    line = build_microphone_line(2, DEVICES[2], selected_index=2)
    assert line.endswith("[selected]")


def test_list_microphones_includes_verification_status():
    stream = StringIO()
    sounddevice = FakeSoundDevice(DEVICES)
    with patch("sys.stdout", stream):
        list_microphones("2", True, sounddevice)
    lines = stream.getvalue().splitlines()
    assert lines == [
        "[0] Built-in Audio (inputs=2, default_sr=48000.0) [ok]",
        "[2] USB Camera (inputs=1, default_sr=44100.0) [selected] [ok]",
    ]


def test_verify_input_device_opens_and_closes_stream():
    sounddevice = FakeSoundDevice(DEVICES)
    index, device = verify_input_device("0", DEVICES, sounddevice)
    assert (index, device) == (0, DEVICES[0])
    assert sounddevice.checked[0]["device"] == 0
    assert sounddevice.stream.started
    assert sounddevice.stream.stopped
    assert sounddevice.stream.closed


def test_is_blank_text_detects_blank_audio_token():
    assert is_blank_text("[BLANK_AUDIO]")
    assert not is_blank_text("hello world")


def test_stop_recording_keeps_sample_rate_for_next_take():
    args = SimpleNamespace(
        model_name="whisper_tiny_en",
        max_tokens=8,
        input_device="0",
    )
    app = PushToTalk(args)
    app.device_index = 0
    app.device = DEVICES[0]
    app.sample_rate = 32000
    app.recording = True
    app.stream = FakeStream()
    app.chunks = [np.array([0.1, 0.2, 0.3], dtype="float32")]
    app.transcribe = lambda waveform, sample_rate: "hello"
    app.stop_recording()
    assert app.sample_rate == 32000
    assert not app.recording
    assert not app.transcribing


class FakeStream:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FakeSoundDevice:
    def __init__(self, devices):
        self.devices = devices
        self.checked = []
        self.stream = None

    def query_devices(self, device=None, kind=None):
        if kind == "input":
            return self.devices[0]
        return self.devices

    def check_input_settings(self, **kwargs):
        self.checked.append(kwargs)

    def InputStream(self, **kwargs):
        del kwargs
        self.stream = FakeStream()
        return self.stream

    def sleep(self, _milliseconds):
        return None
