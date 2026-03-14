import numpy as np
from keras import ops


def allocate_mel_filters(num_mels, num_fft_bins):
    num_nonnegative_fft_bins = 1 + num_fft_bins // 2
    return np.zeros((num_mels, num_nonnegative_fft_bins), dtype=np.float32)


def compute_nonnegative_fft_frequencies(num_fft_bins, sampling_rate):
    return np.fft.rfftfreq(n=num_fft_bins, d=1.0 / sampling_rate)


def build_mel_grid(num_mels, min_mel, max_mel):
    return np.linspace(min_mel, max_mel, num_mels + 2)


def compute_switch_point(min_log_hz, min_hz, linear_hz_per_mel):
    return (min_log_hz - min_hz) / linear_hz_per_mel


def mel_to_hz(mel_values):
    min_hz = 0.0
    linear_hz_per_mel = 200.0 / 3.0
    min_log_hz = 1000.0
    log_step = np.log(6.4) / 27.0
    mel_values = np.asarray(mel_values, dtype=np.float64)
    min_log_mel = compute_switch_point(min_log_hz, min_hz, linear_hz_per_mel)
    frequencies = min_hz + linear_hz_per_mel * mel_values
    use_log_region = mel_values >= min_log_mel
    frequencies[use_log_region] = min_log_hz * np.exp(
        log_step * (mel_values[use_log_region] - min_log_mel)
    )
    return frequencies


def compute_mel_frequency_gaps(mel_frequencies):
    return np.diff(mel_frequencies)


def compute_mel_minus_fft_matrix(mel_frequencies, fft_frequencies):
    return np.subtract.outer(mel_frequencies, fft_frequencies)


def build_single_mel_filter(band_index, mel_minus_fft_matrix, mel_frequency_gaps):
    rising_slope = -mel_minus_fft_matrix[band_index]
    rising_slope = rising_slope / mel_frequency_gaps[band_index]
    fading_slope = mel_minus_fft_matrix[band_index + 2]
    fading_slope = fading_slope / mel_frequency_gaps[band_index + 1]
    return np.maximum(0.0, np.minimum(rising_slope, fading_slope))


def compute_filter_normalization(mel_frequencies, num_mels):
    left_edges = mel_frequencies[:num_mels]
    right_edges = mel_frequencies[2 : num_mels + 2]
    return 2.0 / (right_edges - left_edges)


def build_mel_filters(num_mels, num_fft_bins, sampling_rate, dtype):
    mel_filters = allocate_mel_filters(num_mels, num_fft_bins)
    fft_frequencies = compute_nonnegative_fft_frequencies(
        num_fft_bins, sampling_rate
    )
    mel_grid = build_mel_grid(num_mels, 0.0, 45.245640471924965)
    mel_frequencies = mel_to_hz(mel_grid)
    mel_frequency_gaps = compute_mel_frequency_gaps(mel_frequencies)
    mel_minus_fft_matrix = compute_mel_minus_fft_matrix(
        mel_frequencies,
        fft_frequencies,
    )
    for band_index in range(num_mels):
        mel_filters[band_index] = build_single_mel_filter(
            band_index,
            mel_minus_fft_matrix,
            mel_frequency_gaps,
        )
    normalization = compute_filter_normalization(mel_frequencies, num_mels)
    mel_filters = mel_filters * normalization[:, np.newaxis]
    return np.asarray(mel_filters.T, dtype=dtype)


def mel_spectrogram(inputs, mel_filters):
    return ops.matmul(inputs, mel_filters)
