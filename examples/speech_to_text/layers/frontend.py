import numpy as np
from keras import ops


def frontend(waveform, mel_filters, num_samples, fft_bins, stride):
    waveform, do_squeeze = batch_tensor(waveform)
    waveform = build_fixed_length_waveform(waveform, num_samples)
    waveform = build_stft_waveform(waveform, fft_bins)
    real, imag = compute_stft(waveform, fft_bins, stride)
    power = compute_power_spectrogram(real, imag)
    mel = mel_spectrogram(power, mel_filters)
    features = compute_log_mel_features(mel)
    return squeeze_batch(features, do_squeeze)


def batch_tensor(waveform):
    rank = len(waveform.shape)
    if rank == 1:
        return ops.expand_dims(waveform, axis=0), True
    if rank == 2:
        return waveform, False
    raise ValueError("Audio expects rank 1 or 2 input.")


def build_fixed_length_waveform(waveform, num_samples):
    waveform = ops.pad(waveform, [[0, 0], [0, num_samples]])
    return waveform[:, :num_samples]


def build_stft_waveform(waveform, fft_bins):
    pad_width = fft_bins // 2
    padding = [[0, 0], [pad_width, pad_width]]
    return ops.pad(waveform, padding, mode="reflect")


def compute_stft(waveform, fft_bins, stride):
    kwargs = {"sequence_length": fft_bins,
              "sequence_stride": stride,
              "fft_length": fft_bins,
              "window": "hann", "center": False}
    return ops.stft(waveform, **kwargs)


def compute_power_spectrogram(real_part, imaginary_part):
    real_part = real_part[:, :-1, :]
    imaginary_part = imaginary_part[:, :-1, :]
    return ops.square(real_part) + ops.square(imaginary_part)


def mel_spectrogram(inputs, mel_filters):
    return ops.matmul(inputs, mel_filters)


def compute_log_mel_features(mel_features):
    minimum = ops.cast(1e-10, mel_features.dtype)
    mel_features = ops.maximum(mel_features, minimum)
    log_base = ops.cast(np.log(10.0), mel_features.dtype)
    log_spec = ops.log(mel_features) / log_base
    max_val = ops.max(log_spec, axis=(1, 2), keepdims=True)
    floor = max_val - ops.cast(8.0, log_spec.dtype)
    log_spec = ops.maximum(log_spec, floor)
    scale = ops.cast(4.0, log_spec.dtype)
    return (log_spec + scale) / scale


def squeeze_batch(features, do_squeeze):
    if do_squeeze:
        return ops.squeeze(features, axis=0)
    return features


def build_mel_filters(num_mels, num_fft_bins, sampling_rate, max_mel):
    filters = allocate_mel_filters(num_mels, num_fft_bins)
    fft_freqs = compute_fft_frequencies(num_fft_bins, sampling_rate)
    mel_grid = build_mel_grid(num_mels, 0.0, max_mel)
    mel_freqs = mel_to_hz(mel_grid)
    mel_gaps = np.diff(mel_freqs)
    mel_fft = np.subtract.outer(mel_freqs, fft_freqs)
    for mel_index in range(num_mels):
        args = (mel_index, mel_fft, mel_gaps)
        filters[mel_index] = build_single_mel_filter(*args)
    norm = compute_filter_normalization(mel_freqs, num_mels)
    filters = filters * norm[:, np.newaxis]
    return np.asarray(filters.T)


def allocate_mel_filters(num_mels, num_fft_bins):
    num_bins = 1 + num_fft_bins // 2
    return np.zeros((num_mels, num_bins), dtype=np.float32)


def compute_fft_frequencies(num_fft_bins, sampling_rate):
    return np.fft.rfftfreq(n=num_fft_bins, d=1.0 / sampling_rate)


def build_mel_grid(num_mels, min_mel, max_mel):
    return np.linspace(min_mel, max_mel, num_mels + 2)


def mel_to_hz(mel_values):
    hz_per_mel = 200.0 / 3.0
    min_log_hz, log_step = 1000.0, np.log(6.4) / 27.0
    mel_values = np.asarray(mel_values, dtype=np.float64)
    min_log_mel = (min_log_hz - 0.0) / hz_per_mel
    frequencies = hz_per_mel * mel_values
    log_region = mel_values >= min_log_mel
    log_mels = mel_values[log_region] - min_log_mel
    frequencies[log_region] = min_log_hz * np.exp(log_step * log_mels)
    return frequencies


def build_single_mel_filter(index, mel_fft, mel_gaps):
    rising = -mel_fft[index] / mel_gaps[index]
    fading = mel_fft[index + 2] / mel_gaps[index + 1]
    return np.maximum(0.0, np.minimum(rising, fading))


def compute_filter_normalization(mel_freqs, num_mels):
    left = mel_freqs[:num_mels]
    right = mel_freqs[2:num_mels + 2]
    return 2.0 / (right - left)
