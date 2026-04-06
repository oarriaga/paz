from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from examples.speech_to_text.audio import load
from examples.speech_to_text.audio import resample
from examples.speech_to_text.audio import to_float
from examples.speech_to_text.audio import to_mono


def test_load_returns_expected_waveform_shape(tmp_path):
    wav_path = tmp_path / "mono.wav"
    waveform = np.array([0.0, 0.25, -0.25], dtype="float32")
    wavfile.write(wav_path, 16000, waveform)
    loaded_waveform, _ = load(wav_path)
    assert loaded_waveform.shape == (3,)


def test_load_returns_expected_sample_rate(tmp_path):
    wav_path = tmp_path / "mono.wav"
    waveform = np.array([0.0, 0.25, -0.25], dtype="float32")
    wavfile.write(wav_path, 16000, waveform)
    _, sample_rate = load(wav_path)
    assert sample_rate == 16000


def test_load_reads_harvard_wav_with_extra_chunks():
    wav_path = Path(__file__).with_name("harvard.wav")
    _, sample_rate = load(wav_path)
    assert sample_rate == 44100


def test_to_float_returns_float32_for_float_input():
    waveform = np.array([0.0, 0.25, -0.25], dtype="float64")
    float_waveform = to_float(waveform)
    assert float_waveform.dtype == np.float32


def test_to_float_normalizes_int16_samples():
    waveform = np.array([-32768, 0, 32767], dtype=np.int16)
    float_waveform = to_float(waveform)
    np.testing.assert_allclose(
        float_waveform,
        np.array([-1.0, 0.0, 32767.0 / 32768.0], dtype="float32"),
    )


def test_to_float_normalizes_uint8_samples():
    waveform = np.array([0, 128, 255], dtype=np.uint8)
    float_waveform = to_float(waveform)
    np.testing.assert_allclose(
        float_waveform,
        np.array([-1.0, 0.0, 127.0 / 128.0], dtype="float32"),
    )


def test_to_float_rejects_unsupported_dtype():
    waveform = np.array([True, False], dtype=np.bool_)
    with pytest.raises(ValueError, match="Unsupported WAV dtype"):
        to_float(waveform)


def test_to_mono_preserves_rank_1_waveform():
    waveform = np.array([0.0, 0.25, -0.25], dtype="float32")
    mono_waveform = to_mono(waveform)
    assert mono_waveform.shape == (3,)


def test_to_mono_collapses_single_channel_axis():
    waveform = np.array([[0.0], [0.25], [-0.25]], dtype="float32")
    mono_waveform = to_mono(waveform)
    assert mono_waveform.shape == (3,)


def test_to_mono_averages_stereo_channels():
    waveform = np.array([[0.0, 0.1], [0.25, -0.1], [-0.25, 0.2]], dtype="float32")
    mono_waveform = to_mono(waveform)
    np.testing.assert_allclose(
        mono_waveform,
        np.array([0.05, 0.075, -0.025], dtype="float32"),
    )


def test_to_mono_rejects_multichannel_audio():
    waveform = np.ones((3, 3), dtype="float32")
    with pytest.raises(ValueError, match="Expected WAV input"):
        to_mono(waveform)


def test_resample_keeps_float32_when_sample_rates_match():
    waveform = np.array([0.0, 0.25, -0.25], dtype="float64")
    resampled_waveform = resample(waveform, 16000, 16000)
    assert resampled_waveform.dtype == np.float32


def test_resample_changes_length_when_sample_rates_differ():
    waveform = np.array([0.0, 0.25, -0.25], dtype="float32")
    resampled_waveform = resample(waveform, 8000, 16000)
    assert len(resampled_waveform) == 6
