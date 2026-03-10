import warnings
from math import gcd

import numpy as np
from keras import ops
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
from scipy.signal import resample_poly


def load_waveform_from_wav(wav_path, expected_sample_rate=16000):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", WavFileWarning)
        sample_rate, waveform = wavfile.read(wav_path)
    waveform = build_float_waveform(waveform)
    waveform = build_mono_waveform(waveform)
    waveform = build_resampled_waveform(waveform, sample_rate, expected_sample_rate)  # fmt: skip
    waveform = validate_whisper_waveform(waveform)
    waveform = ops.convert_to_tensor(waveform, dtype="float32")
    return expected_sample_rate, waveform


def validate_whisper_waveform(waveform):
    waveform = np.asarray(waveform, dtype="float32")
    if len(waveform.shape) == 1:
        return np.clip(waveform, -1.0, 1.0)
    raise ValueError("Expected mono waveform after audio preprocessing.")


def build_mono_waveform(waveform):
    if len(waveform.shape) == 1:
        return waveform
    if len(waveform.shape) == 2 and waveform.shape[1] == 1:
        return waveform[:, 0]
    if len(waveform.shape) == 2 and waveform.shape[1] == 2:
        return np.mean(waveform, axis=1)
    raise ValueError("Expected WAV input with shape (N,), (N, 1), or (N, 2).")


def build_resampled_waveform(waveform, sample_rate, expected_sample_rate):
    if sample_rate == expected_sample_rate:
        return waveform.astype("float32")
    common_divisor = gcd(sample_rate, expected_sample_rate)
    upsample = expected_sample_rate // common_divisor
    downsample = sample_rate // common_divisor
    waveform = resample_poly(waveform, upsample, downsample)
    return waveform.astype("float32")


def build_float_waveform(waveform):
    waveform = np.asarray(waveform)
    if np.issubdtype(waveform.dtype, np.floating):
        return waveform.astype("float32")
    if np.issubdtype(waveform.dtype, np.signedinteger):
        scale = max(abs(np.iinfo(waveform.dtype).min), np.iinfo(waveform.dtype).max)  # fmt: skip
        waveform = waveform.astype("float32") / float(scale)
        return np.clip(waveform, -1.0, 1.0)
    if np.issubdtype(waveform.dtype, np.unsignedinteger):
        info = np.iinfo(waveform.dtype)
        midpoint = (info.max + 1) / 2.0
        waveform = waveform.astype("float32")
        waveform = (waveform - midpoint) / midpoint
        return np.clip(waveform, -1.0, 1.0)
    raise ValueError("Unsupported WAV dtype: {}".format(waveform.dtype))
