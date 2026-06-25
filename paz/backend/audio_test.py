import numpy as np

from paz.backend.audio import to_float, to_mono, resample


def test_to_float_int16_full_scale():
    waveform = np.array([-32768, 0, 32767], dtype="int16")
    result = to_float(waveform)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [-1.0, 0.0, 1.0], atol=1e-4)


def test_to_float_passes_through_floating():
    waveform = np.array([-0.5, 0.0, 0.5], dtype="float64")
    result = to_float(waveform)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [-0.5, 0.0, 0.5])


def test_to_mono_averages_stereo():
    waveform = np.array([[1.0, 3.0], [2.0, 4.0]], dtype="float32")
    np.testing.assert_allclose(to_mono(waveform), [2.0, 3.0])


def test_to_mono_keeps_single_channel():
    waveform = np.arange(4, dtype="float32")
    np.testing.assert_allclose(to_mono(waveform), waveform)


def test_resample_changes_length_by_ratio():
    waveform = np.zeros(16000, dtype="float32")
    result = resample(waveform, 16000, 8000)
    assert result.dtype == np.float32
    assert result.shape[0] == 8000


def test_resample_keeps_rate_when_equal():
    waveform = np.linspace(-1.0, 1.0, 100, dtype="float32")
    result = resample(waveform, 16000, 16000)
    np.testing.assert_allclose(result, waveform)
