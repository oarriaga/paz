import numpy as np
import pytest
from keras import ops
from scipy.io import wavfile

from examples.speech_to_text.audio import build_float_waveform
from examples.speech_to_text.audio import load_waveform_from_wav


def test_load_waveform_from_wav_preserves_mono_shape(tmp_path):
    wav_path = tmp_path / "mono.wav"
    waveform = np.array([0.0, 0.25, -0.25], dtype="float32")
    wavfile.write(wav_path, 16000, waveform)
    _, loaded_waveform = load_waveform_from_wav(wav_path)
    assert tuple(loaded_waveform.shape) == (3,)


def test_load_waveform_from_wav_returns_float32(tmp_path):
    wav_path = tmp_path / "float32.wav"
    waveform = np.array([0.0, 0.25, -0.25], dtype="float32")
    wavfile.write(wav_path, 16000, waveform)
    _, loaded_waveform = load_waveform_from_wav(wav_path)
    assert loaded_waveform.dtype == "float32"


def test_load_waveform_from_wav_collapses_single_channel_axis(tmp_path):
    wav_path = tmp_path / "single_channel.wav"
    waveform = np.array([[0.0], [0.25], [-0.25]], dtype="float32")
    wavfile.write(wav_path, 16000, waveform)
    _, loaded_waveform = load_waveform_from_wav(wav_path)
    assert tuple(loaded_waveform.shape) == (3,)


def test_load_waveform_from_wav_resamples_audio(tmp_path):
    wav_path = tmp_path / "resampled.wav"
    waveform = np.array([0.0, 0.25, -0.25], dtype="float32")
    wavfile.write(wav_path, 8000, waveform)
    sample_rate, loaded_waveform = load_waveform_from_wav(wav_path)
    assert (sample_rate, len(loaded_waveform)) == (16000, 6)


def test_load_waveform_from_wav_averages_stereo_channels(tmp_path):
    wav_path = tmp_path / "stereo.wav"
    waveform = np.array([[0.0, 0.1], [0.25, -0.1], [-0.25, 0.2]], dtype="float32")
    wavfile.write(wav_path, 16000, waveform)
    _, loaded_waveform = load_waveform_from_wav(wav_path)
    np.testing.assert_allclose(
        ops.convert_to_numpy(loaded_waveform),
        np.array([0.05, 0.075, -0.025], dtype="float32"),
    )


def test_load_waveform_from_wav_rejects_multichannel_audio(tmp_path):
    wav_path = tmp_path / "multichannel.wav"
    waveform = np.ones((3, 3), dtype="float32")
    wavfile.write(wav_path, 16000, waveform)
    with pytest.raises(ValueError, match="Expected WAV input"):
        load_waveform_from_wav(wav_path)


def test_build_float_waveform_normalizes_int16_samples():
    waveform = np.array([-32768, 0, 32767], dtype=np.int16)
    normalized_waveform = build_float_waveform(waveform)
    normalized_waveform = ops.convert_to_numpy(normalized_waveform)
    np.testing.assert_allclose(
        normalized_waveform,
        np.array([-1.0, 0.0, 32767.0 / 32768.0], dtype="float32"),
    )
