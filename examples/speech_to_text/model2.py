import string
from pathlib import Path

import numpy as np
import keras
from keras.activations import gelu
from keras import ops
from keras import Model
from keras.layers import Input, Conv1D, Dropout, Add, Lambda, LayerNormalization
from keras.layers import Embedding, Dense, EinsumDense, ReversibleEmbedding


WHISPER_MODELS_DIR = Path(__file__).resolve().with_name("whisper_models")


CONFIGS = {
    "whisper_tiny_en": {
        "vocabulary_size": 51864,
        "num_layers": 4,
        "num_heads": 6,
        "hidden_dim": 384,
        "intermediate_dim": 1536,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_base_en": {
        "vocabulary_size": 51864,
        "num_layers": 6,
        "num_heads": 8,
        "hidden_dim": 512,
        "intermediate_dim": 2048,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_small_en": {
        "vocabulary_size": 51864,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "intermediate_dim": 3072,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_medium_en": {
        "vocabulary_size": 51864,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "intermediate_dim": 4096,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_tiny_multi": {
        "vocabulary_size": 51865,
        "num_layers": 4,
        "num_heads": 6,
        "hidden_dim": 384,
        "intermediate_dim": 1536,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_base_multi": {
        "vocabulary_size": 51865,
        "num_layers": 6,
        "num_heads": 8,
        "hidden_dim": 512,
        "intermediate_dim": 2048,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_small_multi": {
        "vocabulary_size": 51865,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "intermediate_dim": 3072,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_medium_multi": {
        "vocabulary_size": 51865,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "intermediate_dim": 4096,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_large_multi": {
        "vocabulary_size": 51865,
        "num_layers": 32,
        "num_heads": 20,
        "hidden_dim": 1280,
        "intermediate_dim": 5120,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_large_multi_v2": {
        "vocabulary_size": 51865,
        "num_layers": 32,
        "num_heads": 20,
        "hidden_dim": 1280,
        "intermediate_dim": 5120,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
}


def build_whisper_model_dir(variant_name):
    return WHISPER_MODELS_DIR / variant_name


def build_whisper_weights_path(variant_name, model_kind):
    return build_whisper_model_dir(variant_name) / "{}.weights.h5".format(model_kind)


def load_serialized_whisper_weights(model, variant_name, model_kind):
    weights_path = build_whisper_weights_path(variant_name, model_kind)
    if not weights_path.exists():
        message = "No serialized {} weights found for {} at {}.".format(model_kind, variant_name, weights_path)  # fmt: skip
        message += " Run export_whisper_models.py to port the local preset into standard Keras files."  # fmt: skip
        raise FileNotFoundError(message)
    model.load_weights(str(weights_path))
    return model


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
    return ops.stft(waveform, sequence_length=num_fft_bins, sequence_stride=stride, fft_length=num_fft_bins, window="hann", center=False)  # fmt: skip


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


def frontend(waveform, mel_filters, num_fft_bins=400, stride=160, sampling_rate=16000, max_audio_length=30):  # fmt: skip
    num_samples = sampling_rate * max_audio_length
    waveform = ops.cast(waveform, "float32")
    waveform, squeeze_batch_axis = batch_tensor(waveform)
    waveform = build_fixed_length_waveform(waveform, num_samples)
    waveform = build_stft_waveform(waveform, num_fft_bins)
    real_part, imaginary_part = compute_stft_components(waveform, num_fft_bins, stride)  # fmt: skip
    power_spectrogram = compute_power_spectrogram(real_part, imaginary_part)
    mel_features = mel_spectrogram(power_spectrogram, mel_filters)
    features = compute_log_mel_features(mel_features)
    return squeeze_features_batch_axis(features, squeeze_batch_axis)


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
    frequencies[use_log_region] = min_log_hz * np.exp(log_step * (mel_values[use_log_region] - min_log_mel))  # fmt: skip
    return frequencies


def compute_mel_frequency_gaps(mel_frequencies):
    return np.diff(mel_frequencies)


def compute_mel_minus_fft_matrix(mel_frequencies, fft_frequencies):
    return np.subtract.outer(mel_frequencies, fft_frequencies)


def build_single_mel_filter(band_index, mel_minus_fft_matrix, mel_frequency_gaps):  # fmt: skip
    rising_slope = -mel_minus_fft_matrix[band_index]
    rising_slope = rising_slope / mel_frequency_gaps[band_index]
    fading_slope = mel_minus_fft_matrix[band_index + 2]
    fading_slope = fading_slope / mel_frequency_gaps[band_index + 1]
    return np.maximum(0.0, np.minimum(rising_slope, fading_slope))


def compute_filter_normalization(mel_frequencies, num_mels):
    left_edges = mel_frequencies[:num_mels]
    right_edges = mel_frequencies[2 : num_mels + 2]  # fmt: skip
    return 2.0 / (right_edges - left_edges)


def build_mel_filters(num_mels, num_fft_bins, sampling_rate):
    mel_filters = allocate_mel_filters(num_mels, num_fft_bins)
    fft_frequencies = compute_nonnegative_fft_frequencies(num_fft_bins, sampling_rate)  # fmt: skip
    mel_grid = build_mel_grid(num_mels, 0.0, 45.245640471924965)
    mel_frequencies = mel_to_hz(mel_grid)
    mel_frequency_gaps = compute_mel_frequency_gaps(mel_frequencies)
    mel_minus_fft_matrix = compute_mel_minus_fft_matrix(mel_frequencies, fft_frequencies)  # fmt: skip
    for band_index in range(num_mels):
        mel_filters[band_index] = build_single_mel_filter(band_index, mel_minus_fft_matrix, mel_frequency_gaps)  # fmt: skip
    normalization = compute_filter_normalization(mel_frequencies, num_mels)
    mel_filters = mel_filters * normalization[:, np.newaxis]
    return np.asarray(mel_filters.T)


def mel_spectrogram(inputs, mel_filters):
    return ops.matmul(inputs, mel_filters)


def WhisperFrontend(num_mels=80, num_fft_bins=400, stride=160, sampling_rate=16000, max_audio_length=30, dtype="float32", name="whisper_frontend"):  # fmt: skip
    waveform = Input(shape=(None,), dtype="float32", name="waveform")
    mel_filters = build_mel_filters(num_mels, num_fft_bins, sampling_rate)
    mel_filters = ops.convert_to_tensor(mel_filters, dtype=dtype)
    features = frontend(waveform, mel_filters, num_fft_bins, stride, sampling_rate, max_audio_length)  # fmt: skip
    return Model(waveform, features, name=name)


def WhisperEncoder(vocabulary_size, num_layers, num_heads, hidden_dim, intermediate_dim, num_mels=80, dropout=0.0, max_encoder_sequence_length=3000, max_decoder_sequence_length=448, dtype="float32", name="whisper_encoder", weights=None):  # fmt: skip
    encoder_features = Input(shape=(None, num_mels), dtype="float32", name="encoder_features")
    conv_name = "encoder_token_embedding_conv_layer"
    x = Conv1D(hidden_dim, 3, 1, "same", dtype=dtype, name=f"{conv_name}_1")(encoder_features)
    x = gelu(x, approximate=False)
    x = Lambda(lambda t: ops.pad(t, [[0, 0], [1, 1], [0, 0]]), output_shape=lambda shape: (shape[0], None, shape[2]), name="encoder_padder")(x)  # fmt: skip
    x = Conv1D(hidden_dim, 3, 2, "valid", dtype=dtype, name=f"{conv_name}_2")(x)
    x = gelu(x, approximate=False)
    positions = embed_position(x, max_encoder_sequence_length // 2, False, None, "encoder_position_embedding")  # fmt: skip
    x = Add(dtype=dtype, name="encoder_embeddings_add")((x, positions))
    x = Dropout(dropout, dtype=dtype, name="encoder_embeddings_dropout")(x)
    for layer_index in range(num_layers):
        layer_name = "transformer_encoder_layer_{}".format(layer_index)
        x = encoder_block(x, num_heads, intermediate_dim, layer_name, dropout, 1e-5)
    y = LayerNormalization(axis=-1, epsilon=1e-5, dtype=dtype, name="encoder_layer_norm")(x)
    model = Model(encoder_features, y, name=name)

    if weights is not None:
        load_serialized_whisper_weights(model, weights, "encoder")
    return model


def build_position_args(sequence_length, start_arg):
    return ops.arange(start_arg, start_arg + sequence_length, dtype="int32")


def broadcast(position_embeddings, inputs):
    return ops.ones_like(inputs) * position_embeddings


def Kernel(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


def self_attend(x, num_heads, key_dim, dropout, epsilon, name):
    delta = LayerNormalization(epsilon=epsilon, name=f"{name}_norm")(x)
    delta = attention(delta, delta, None, None, num_heads, key_dim, None, dropout, True, False, Kernel(), "zeros", "float32", name)  # fmt: skip
    delta = Dropout(dropout, name=f"{name}_dropout")(delta)
    return x + delta


def encoder_dense(x, intermediate_dim, dropout, epsilon, name):
    delta = LayerNormalization(epsilon=epsilon, name=f"{name}_layer_norm")(x)
    delta = Dense(intermediate_dim, activation=gelu, kernel_initializer=Kernel(), bias_initializer="zeros", name=f"{name}_intermediate_dense")(delta)  # fmt: skip
    delta = Dense(x.shape[-1], kernel_initializer=Kernel(), bias_initializer="zeros", name=f"{name}_output_dense")(delta)  # fmt: skip
    delta = Dropout(dropout, name=f"{name}_dropout")(delta)
    return x + delta


def decoder_dense(x, intermediate_dim, dropout, epsilon, name):  # fmt: skip
    delta = LayerNormalization(epsilon=epsilon, name=f"{name}_layer_norm")(x)
    delta = Dense(
        intermediate_dim,
        activation=gelu,
        kernel_initializer=Kernel(),
        bias_initializer="zeros",
        name=f"{name}_intermediate_dense",
    )(delta)
    delta = Dense(
        x.shape[-1],
        kernel_initializer=Kernel(),
        bias_initializer="zeros",
        name=f"{name}_output_dense",
    )(delta)
    delta = Dropout(dropout, name=f"{name}_dropout")(delta)
    return x + delta


def encoder_block(x, num_heads, dim, name, dropout, epsilon):
    key_dim = int(x.shape[-1] // num_heads)
    x = self_attend(x, num_heads, key_dim, dropout, epsilon, f"{name}_self_attention_layer")  # fmt: skip
    return encoder_dense(x, dim, dropout, epsilon, f"{name}_feedforward")


def embed_position(x, sequence_length, trainable, positions, name):
    if positions is None:
        positions = Lambda(lambda x: build_position_args(ops.shape(x)[-2], 0), output_shape=lambda shape: (shape[-2],), name=f"{name}_indices")(x)  # fmt: skip
        feature_size, kwargs = x.shape[-1], {"trainable": trainable, "name": name}  # fmt: skip
        embeddings = Embedding(sequence_length, feature_size, Kernel(), **kwargs)(positions)  # fmt: skip
        embeddings = ops.expand_dims(embeddings, axis=0)
        return broadcast(embeddings, x)
    if len(positions.shape) == 1:
        positions = ops.expand_dims(positions, axis=0)
    feature_size, kwargs = x.shape[-1], {"trainable": trainable, "name": name}  # fmt: skip
    embeddings = Embedding(sequence_length, feature_size, Kernel(), **kwargs)(positions)  # fmt: skip
    return broadcast(embeddings, x)


def build_cached_self_attention_mask(self_attention_cache, cache_update_index):
    valid_positions = ops.ones_like(self_attention_cache[:, 0, :, 0, 0], dtype="int32")  # fmt: skip
    valid_positions = ops.cumsum(valid_positions, axis=1) - 1
    return ops.cast(ops.expand_dims(valid_positions, axis=1) <= cache_update_index, "int32")  # fmt: skip


def cached_self_attention_block(decoder_sequence, self_attention_cache, cache_update_index, num_heads, key_dim, dropout=0.0, epsilon=1e-5, name="self_attention"):  # fmt: skip
    residual = decoder_sequence
    attention_mask = build_cached_self_attention_mask(self_attention_cache, cache_update_index)  # fmt: skip
    hidden = LayerNormalization(epsilon=epsilon, name=f"{name}_layer_norm")(decoder_sequence)  # fmt: skip
    hidden, self_attention_cache = cached_attention(hidden, self_attention_cache, cache_update_index, hidden, None, attention_mask, num_heads, key_dim, None, dropout, True, Kernel(), "zeros", name)  # fmt: skip
    hidden = Dropout(dropout, name=f"{name}_dropout")(hidden)
    return hidden + residual, self_attention_cache


def index_to_einsum_variable(index):
    return string.ascii_lowercase[index]


def build_projection_equation(free_dims, bound_dims, output_dims):
    input_string = ""
    kernel_string = ""
    output_string = ""
    bias_axes = ""
    letter_offset = 0
    for index in range(free_dims):
        character = index_to_einsum_variable(index + letter_offset)
        input_string += character
        output_string += character
    letter_offset += free_dims
    for index in range(bound_dims):
        character = index_to_einsum_variable(index + letter_offset)
        input_string += character
        kernel_string += character
    letter_offset += bound_dims
    for index in range(output_dims):
        character = index_to_einsum_variable(index + letter_offset)
        kernel_string += character
        output_string += character
        bias_axes += character
    equation = f"{input_string},{kernel_string}->{output_string}"
    return equation, bias_axes, len(output_string)


def build_output_shape(output_rank, known_last_dims):
    return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)


def project(x, free_dims, bound_dims, output_dims, output_shape, use_bias, kernel_initializer, bias_initializer, name):  # fmt: skip
    equation, bias_axes, output_rank = build_projection_equation(free_dims, bound_dims, output_dims)  # fmt: skip
    final_shape = build_output_shape(output_rank - 1, output_shape)
    bias_axes = bias_axes if use_bias else None
    values = [bias_axes, kernel_initializer, bias_initializer, name]
    keys = ["bias_axes", "kernel_initializer", "bias_initializer", "name"]
    return EinsumDense(equation, final_shape, **dict(zip(keys, values)))(x)


def expand_attention_mask_for_heads(attention_mask):
    if attention_mask is None:
        return None
    attention_mask = ops.cast(attention_mask, "bool")
    if len(attention_mask.shape) == 2:
        attention_mask = ops.expand_dims(attention_mask, axis=1)
        attention_mask = ops.expand_dims(attention_mask, axis=1)
    elif len(attention_mask.shape) == 3:
        attention_mask = ops.expand_dims(attention_mask, axis=1)
    return attention_mask


def merge_heads_into_last_dimension(tensor):
    return ops.transpose(tensor, (0, 2, 1, 3))


def mask_scores(attention_scores, attention_mask):
    if attention_mask is None:
        return attention_scores
    large_negative_value = ops.cast(-1e9, attention_scores.dtype)
    return ops.where(attention_mask, attention_scores, large_negative_value)


def build_attention_cache(value, key, num_heads, key_dim, value_dim, use_bias, kernel_initializer, bias_initializer, name):  # fmt: skip
    if key is None:
        key = value
    if value_dim is None:
        value_dim = key_dim
    value_rank, key_rank = len(value.shape), len(key.shape)
    args = (False, kernel_initializer, bias_initializer, f"{name}_key")
    key = project(key, key_rank - 1, 1, 2, [num_heads, key_dim], *args)
    args = (use_bias, kernel_initializer, bias_initializer, f"{name}_value")
    value = project(value, value_rank - 1, 1, 2, [num_heads, value_dim], *args)
    return ops.stack((key, value), axis=1)


def cached_attention(query, cache, cache_update_index, value, key, attention_mask, num_heads, key_dim, value_dim, dropout, use_bias, kernel_initializer, bias_initializer, name):  # fmt: skip
    if value_dim is None:
        value_dim = key_dim

    query_rank = len(query.shape)
    output_dim = query.shape[-1]
    query_projection = project(query, query_rank - 1, 1, 2, [num_heads, key_dim], use_bias, kernel_initializer, bias_initializer, f"{name}_query")  # fmt: skip
    key_projection, value_projection = cache[:, 0, ...], cache[:, 1, ...]
    if cache_update_index is not None:
        if value is None:
            raise ValueError("Cached attention updates require value inputs.")
        update_cache = build_attention_cache(value, key, num_heads, key_dim, value_dim, use_bias, kernel_initializer, bias_initializer, name)  # fmt: skip
        start = [0, 0, cache_update_index, 0, 0]
        cache = ops.slice_update(cache, start, update_cache)
        key_projection = cache[:, 0, ...]
        value_projection = cache[:, 1, ...]

    query_heads = ops.transpose(query_projection, (0, 2, 1, 3))
    key_heads = ops.transpose(key_projection, (0, 2, 1, 3))
    value_heads = ops.transpose(value_projection, (0, 2, 1, 3))
    scaling_factor = ops.sqrt(ops.cast(key_dim, query_heads.dtype))
    query_heads = query_heads / scaling_factor
    key_heads = ops.transpose(key_heads, (0, 1, 3, 2))
    scores = ops.matmul(query_heads, key_heads)
    mask = expand_attention_mask_for_heads(attention_mask)
    scores = mask_scores(scores, mask)
    probabilities = ops.softmax(scores, axis=-1)
    probabilities = Dropout(dropout, name=f"{name}_attention_scores_dropout")(probabilities)  # fmt: skip
    values = ops.matmul(probabilities, value_heads)
    values = merge_heads_into_last_dimension(values)
    output = project(values, query_rank - 1, 2, 1, [output_dim], use_bias, kernel_initializer, bias_initializer, f"{name}_attention_output")  # fmt: skip
    return output, cache


def cached_cross_attention_block(x, cross_attention_cache, num_heads, key_dim, dropout=0.0, epsilon=1e-5, name="cross_attention"):  # fmt: skip
    delta = LayerNormalization(epsilon=epsilon, name=f"{name}_layer_norm")(x)
    delta, _ = cached_attention(delta, cross_attention_cache, None, None, None, None, num_heads, key_dim, key_dim, dropout, True, Kernel(), "zeros", name)  # fmt: skip
    delta = Dropout(dropout, name=f"{name}_dropout")(delta)
    return x + delta


def cached_decoder_block(decoder_sequence, self_attention_cache, cross_attention_cache, cache_update_index, num_heads=8, intermediate_dim=2048, dropout=0.0, epsilon=1e-5, name="transformer_decoder_layer"):  # fmt: skip
    hidden_dim = decoder_sequence.shape[-1]
    if hidden_dim is None:
        raise ValueError("Decoder block inputs must have known hidden size.")
    key_dim = int(hidden_dim // num_heads)
    hidden, self_attention_cache = cached_self_attention_block(decoder_sequence, self_attention_cache, cache_update_index, num_heads, key_dim, dropout, epsilon, f"{name}_self_attention")  # fmt: skip
    hidden = cached_cross_attention_block(hidden, cross_attention_cache, num_heads, key_dim, dropout, epsilon, f"{name}_cross_attention")  # fmt: skip
    hidden = decoder_dense(hidden, intermediate_dim, dropout, epsilon, f"{name}_feedforward")  # fmt: skip
    return hidden, self_attention_cache


def WhisperDecoderStep(vocabulary_size, num_layers, num_heads, hidden_dim, intermediate_dim, num_mels=80, dropout=0.0, max_encoder_sequence_length=3000, max_decoder_sequence_length=448, dtype="float32", name="whisper_decoder_step", weights=None):
    key_dim = int(hidden_dim // num_heads)
    decoder_token_ids = Input((1,), dtype="int32", name="decoder_token_ids")  # fmt: skip
    self_attention_cache = Input((num_layers, 2, None, num_heads, key_dim), dtype="float32", name="self_attention_cache")  # fmt: skip
    cross_attention_cache = Input((num_layers, 2, None, num_heads, key_dim), dtype="float32", name="cross_attention_cache")  # fmt: skip
    cache_update_index = Input((), dtype="int32", name="cache_update_index")  # fmt: skip
    cache_update_index_scalar = Lambda(
        lambda x: ops.cast(x[0], "int32"),
        output_shape=(),
        name="cache_update_index_scalar",
    )(cache_update_index)

    positions = Lambda(
        lambda x: ops.expand_dims(ops.cast(x, "int32"), axis=-1),
        output_shape=(1,),
        name="decoder_position_indices",
    )(cache_update_index)

    decoder_embeddings = ReversibleEmbedding( vocabulary_size, hidden_dim, tie_weights=True, embeddings_initializer=Kernel(), mask_zero=False, name="decoder_token_embedding")  # fmt: skip
    hidden = decoder_embeddings(decoder_token_ids)
    position_embeddings = embed_position(hidden, max_decoder_sequence_length, True, positions, "decoder_position_embedding")
    hidden = Add(dtype=dtype, name="decoder_embeddings_add")((hidden, position_embeddings))  # fmt: skip
    hidden = Dropout(dropout, dtype=dtype, name="decoder_embeddings_dropout")(hidden)  # fmt: skip
    updated_self_attention_caches = []
    for layer_index in range(num_layers):
        layer_self_attention_cache = Lambda(
            lambda x, index: x[:, index, ...],
            output_shape=(2, None, num_heads, key_dim),
            arguments={"index": layer_index},
            name="transformer_decoder_layer_{}_self_attention_cache_slice".format(
                layer_index
            ),
        )(self_attention_cache)
        layer_cross_attention_cache = Lambda(
            lambda x, index: x[:, index, ...],
            output_shape=(2, None, num_heads, key_dim),
            arguments={"index": layer_index},
            name="transformer_decoder_layer_{}_cross_attention_cache_slice".format(
                layer_index
            ),
        )(cross_attention_cache)
        hidden, layer_self_attention_cache = cached_decoder_block(
            hidden,
            layer_self_attention_cache,
            layer_cross_attention_cache,
            cache_update_index_scalar,
            num_heads,
            intermediate_dim,
            dropout,
            1e-5,
            "transformer_decoder_layer_{}".format(layer_index),
        )
        layer_self_attention_cache = Lambda(
            lambda x: ops.expand_dims(x, axis=1),
            output_shape=(1, 2, None, num_heads, key_dim),
            name="transformer_decoder_layer_{}_self_attention_cache_expand".format(
                layer_index
            ),
        )(layer_self_attention_cache)
        updated_self_attention_caches.append(layer_self_attention_cache)
    updated_self_attention_cache = Lambda(
        lambda tensors: ops.concatenate(tensors, axis=1),
        output_shape=(num_layers, 2, None, num_heads, key_dim),
        name="updated_self_attention_cache",
    )(updated_self_attention_caches)
    hidden = LayerNormalization(axis=-1, epsilon=1e-5, dtype=dtype, name="decoder_layer_norm")(hidden)  # fmt: skip
    logits = decoder_embeddings(hidden, reverse=True)
    model = Model([decoder_token_ids, self_attention_cache, cross_attention_cache, cache_update_index], [logits, updated_self_attention_cache], name=name)  # fmt: skip
    if weights is not None:
        load_serialized_whisper_weights(model, weights, "decoder_step")
    return model


def WhisperCrossCache(vocabulary_size, num_layers, num_heads, hidden_dim, intermediate_dim, num_mels=80, dropout=0.0, max_encoder_sequence_length=3000, max_decoder_sequence_length=448, dtype="float32", name="whisper_cross_cache", weights=None):
    encoder_output = Input(shape=(None, hidden_dim), dtype="float32", name="encoder_output")  # fmt: skip
    key_dim = int(hidden_dim // num_heads)
    caches = []
    for layer_index in range(num_layers):
        cache = build_attention_cache(encoder_output, None, num_heads, key_dim, key_dim, True, Kernel(), "zeros", "transformer_decoder_layer_{}_cross_attention".format(layer_index))  # fmt: skip
        cache = Lambda(
            lambda x: ops.expand_dims(x, axis=1),
            output_shape=(1, 2, None, num_heads, key_dim),
            name="transformer_decoder_layer_{}_cross_attention_cache_expand".format(
                layer_index
            ),
        )(cache)
        caches.append(cache)
    cross_attention_cache = Lambda(
        lambda tensors: ops.concatenate(tensors, axis=1),
        output_shape=(num_layers, 2, None, num_heads, key_dim),
        name="cross_attention_cache",
    )(caches)
    model = Model(encoder_output, cross_attention_cache, name=name)
    if weights is not None:
        load_serialized_whisper_weights(model, weights, "cross_cache")
    return model


def build_padding_attention_mask(padding_mask, attention_mask=None):
    mask = None
    if padding_mask is not None:
        mask = ops.cast(ops.expand_dims(padding_mask, axis=1), "int32")
    if attention_mask is None:
        return mask
    attention_mask = ops.cast(attention_mask, "int32")
    if mask is None:
        return attention_mask
    return ops.minimum(mask, attention_mask)


def build_causal_attention_mask(query, value):
    query_positions = ops.ones_like(query[..., 0], dtype="int32")
    key_positions = ops.ones_like(value[..., 0], dtype="int32")
    query_positions = ops.cumsum(query_positions, axis=1)
    key_positions = ops.cumsum(key_positions, axis=1)
    return ops.cast(
        ops.expand_dims(query_positions, axis=2)
        >= ops.expand_dims(key_positions, axis=1),
        "int32",
    )


def build_combined_attention_mask(attention_mask, use_causal_mask, query, value):
    padding_mask = expand_attention_mask_for_heads(attention_mask)
    if not use_causal_mask:
        return padding_mask
    causal_mask = build_causal_attention_mask(query, value)
    causal_mask = expand_attention_mask_for_heads(causal_mask)
    if padding_mask is None:
        return causal_mask
    return ops.logical_and(padding_mask, causal_mask)


def attention(query, value, key=None, attention_mask=None, num_heads=8, key_dim=64, value_dim=None, dropout=0.0, use_bias=True, use_causal_mask=False, kernel_initializer=None, bias_initializer="zeros", dtype="float32", name="attention"):  # fmt: skip
    if key is None:
        key = value
    if value_dim is None:
        value_dim = key_dim
    if kernel_initializer is None:
        kernel_initializer = Kernel()
    query_rank = len(query.shape)
    value_rank = len(value.shape)
    key_rank = len(key.shape)
    output_dim = query.shape[-1]
    query_projection = project(query, query_rank - 1, 1, 2, [num_heads, key_dim], use_bias, kernel_initializer, bias_initializer, f"{name}_query")  # fmt: skip
    key_projection = project(key, key_rank - 1, 1, 2, [num_heads, key_dim], False, kernel_initializer, bias_initializer, f"{name}_key")  # fmt: skip
    value_projection = project(value, value_rank - 1, 1, 2, [num_heads, value_dim], use_bias, kernel_initializer, bias_initializer, f"{name}_value")  # fmt: skip
    query_heads = ops.transpose(query_projection, (0, 2, 1, 3))
    key_heads = ops.transpose(key_projection, (0, 2, 1, 3))
    value_heads = ops.transpose(value_projection, (0, 2, 1, 3))
    scaling_factor = ops.sqrt(ops.cast(key_dim, query_heads.dtype))
    query_heads = query_heads / scaling_factor
    key_heads = ops.transpose(key_heads, (0, 1, 3, 2))
    attention_scores = ops.matmul(query_heads, key_heads)
    attention_mask = build_combined_attention_mask(attention_mask, use_causal_mask, query, value)
    attention_scores = mask_scores(attention_scores, attention_mask)
    attention_probabilities = ops.softmax(attention_scores, axis=-1)
    attention_probabilities = Dropout(dropout, dtype=dtype, name=f"{name}_attention_scores_dropout")(attention_probabilities)  # fmt: skip
    attention_values = ops.matmul(attention_probabilities, value_heads)
    attention_values = merge_heads_into_last_dimension(attention_values)
    return project(attention_values, query_rank - 1, 2, 1, [output_dim], use_bias, kernel_initializer, bias_initializer, f"{name}_attention_output")  # fmt: skip
