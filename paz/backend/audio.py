import warnings
from math import gcd

import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
from scipy.signal import resample_poly


def load(wav_path):
    # SciPy warns on extra non-audio WAV chunks; keep loading the audio data.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", WavFileWarning)
        sample_rate, waveform = wavfile.read(wav_path)
    return waveform, sample_rate


def to_float(waveform):
    waveform = np.asarray(waveform)
    if np.issubdtype(waveform.dtype, np.floating):
        waveform = waveform.astype("float32")
    elif np.issubdtype(waveform.dtype, np.signedinteger):
        min_value = abs(np.iinfo(waveform.dtype).min)
        max_value = np.iinfo(waveform.dtype).max
        scale = max(min_value, max_value)
        waveform = waveform.astype("float32") / float(scale)
        waveform = np.clip(waveform, -1.0, 1.0)
    elif np.issubdtype(waveform.dtype, np.unsignedinteger):
        info = np.iinfo(waveform.dtype)
        midpoint = (info.max + 1) / 2.0
        waveform = waveform.astype("float32")
        waveform = (waveform - midpoint) / midpoint
        waveform = np.clip(waveform, -1.0, 1.0)
    else:
        raise ValueError("Unsupported WAV dtype: {}".format(waveform.dtype))
    return waveform


def to_mono(waveform):
    if len(waveform.shape) == 2 and waveform.shape[1] == 1:
        waveform = waveform[:, 0]
    elif len(waveform.shape) == 2 and waveform.shape[1] == 2:
        waveform = np.mean(waveform, axis=1)
    elif len(waveform.shape) != 1:
        raise ValueError("Expected WAV input of shape (N,), (N, 1), or (N, 2).")
    return waveform


def resample(waveform, sample_rate, expected_sample_rate):
    if sample_rate != expected_sample_rate:
        common_divisor = gcd(sample_rate, expected_sample_rate)
        upsample = expected_sample_rate // common_divisor
        downsample = sample_rate // common_divisor
        waveform = resample_poly(waveform, upsample, downsample)
    waveform = waveform.astype("float32")
    return waveform
