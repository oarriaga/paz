import numpy as np
from keras import ops

from .mel_spectrogram import mel_spectrogram


def batch_tensor(waveform):
    rank = len(waveform.shape)
    if rank == 1:
        return ops.expand_dims(waveform, axis=0), True
    if rank == 2:
        return waveform, False
    raise ValueError("Audio frontend expects rank 1 or 2 input.")


def build_fixed_length_waveform(waveform, num_samples):
    waveform = ops.pad(waveform, [[0, 0], [0, num_samples]])
    return waveform[:, :num_samples]


def build_stft_waveform(waveform, num_fft_bins):
    pad_width = num_fft_bins // 2
    return ops.pad(waveform, [[0, 0], [pad_width, pad_width]], mode="reflect")


def compute_stft_components(waveform, num_fft_bins, stride):
    return ops.stft(
        waveform,
        sequence_length=num_fft_bins,
        sequence_stride=stride,
        fft_length=num_fft_bins,
        window="hann",
        center=False,
    )


def compute_power_spectrogram(real_part, imaginary_part):
    real_part = real_part[:, :-1, :]
    imaginary_part = imaginary_part[:, :-1, :]
    real_power = ops.square(real_part)
    imaginary_power = ops.square(imaginary_part)
    return real_power + imaginary_power


def compute_log_mel_features(mel_features):
    minimum_value = ops.cast(1e-10, mel_features.dtype)
    mel_features = ops.maximum(mel_features, minimum_value)
    log_spectrogram = ops.log(mel_features)
    log_base = ops.cast(np.log(10.0), log_spectrogram.dtype)
    log_spectrogram = log_spectrogram / log_base
    max_value = ops.max(log_spectrogram, axis=(1, 2), keepdims=True)
    max_value_minus_eight = max_value - ops.cast(8.0, log_spectrogram.dtype)
    log_spectrogram = ops.maximum(log_spectrogram, max_value_minus_eight)
    scale = ops.cast(4.0, log_spectrogram.dtype)
    return (log_spectrogram + scale) / scale


def squeeze_features_batch_axis(features, squeeze_batch_axis):
    if squeeze_batch_axis:
        return ops.squeeze(features, axis=0)
    return features


def frontend(
    waveform,
    mel_filters,
    num_fft_bins=400,
    stride=160,
    sampling_rate=16000,
    max_audio_length=30,
    dtype="float32",
):
    num_samples = sampling_rate * max_audio_length
    waveform = ops.cast(waveform, dtype)
    waveform, squeeze_batch_axis = batch_tensor(waveform)
    waveform = build_fixed_length_waveform(waveform, num_samples)
    waveform = build_stft_waveform(waveform, num_fft_bins)
    real_part, imaginary_part = compute_stft_components(
        waveform,
        num_fft_bins,
        stride,
    )
    power_spectrogram = compute_power_spectrogram(real_part, imaginary_part)
    mel_features = mel_spectrogram(power_spectrogram, mel_filters)
    features = compute_log_mel_features(mel_features)
    return squeeze_features_batch_axis(features, squeeze_batch_axis)
