from io import StringIO

import pytest

from examples.speech_to_text.microphone import collect_input_devices
from examples.speech_to_text.microphone import list_input_devices
from examples.speech_to_text.microphone import resolve_input_device


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
        "name": "USB Camera Microphone",
        "max_input_channels": 1,
        "default_samplerate": 44100.0,
    },
    {
        "name": "USB Camera Pro",
        "max_input_channels": 1,
        "default_samplerate": 44100.0,
    },
]


def test_collect_input_devices_filters_non_input_devices():
    assert [index for index, _ in collect_input_devices(DEVICES)] == [0, 2, 3]


def test_list_input_devices_prints_input_devices_only():
    stream = StringIO()
    list_input_devices(stream, DEVICES)
    assert stream.getvalue().splitlines() == [
        "[0] Built-in Audio (inputs=2, default_sr=48000.0)",
        "[2] USB Camera Microphone (inputs=1, default_sr=44100.0)",
        "[3] USB Camera Pro (inputs=1, default_sr=44100.0)",
    ]


def test_resolve_input_device_uses_default_when_not_configured():
    default_device = {
        "name": "Default Input",
        "max_input_channels": 1,
        "default_samplerate": 16000.0,
    }
    assert resolve_input_device(None, DEVICES, default_device) == (
        None,
        default_device,
    )


def test_resolve_input_device_selects_numeric_index():
    assert resolve_input_device("2", DEVICES) == (2, DEVICES[2])


def test_resolve_input_device_selects_unique_name_match():
    assert resolve_input_device("built-in", DEVICES) == (0, DEVICES[0])


def test_resolve_input_device_rejects_missing_name():
    with pytest.raises(ValueError, match="No input device matches"):
        resolve_input_device("webcam", DEVICES)


def test_resolve_input_device_rejects_ambiguous_name():
    with pytest.raises(ValueError, match="Multiple input devices match"):
        resolve_input_device("usb camera", DEVICES)


def test_resolve_input_device_rejects_non_input_index():
    with pytest.raises(ValueError, match="is not an input device"):
        resolve_input_device("1", DEVICES)
