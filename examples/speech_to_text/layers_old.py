import string

import keras
import numpy as np
from keras import ops
from keras.layers import MultiHeadAttention
from keras.layers import ReversibleEmbedding


def whisper_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


def build_batched_waveform(waveform):
    rank = len(waveform.shape)
    if rank == 1:
        return ops.expand_dims(waveform, axis=0), True
    if rank == 2:
        return waveform, False
    raise ValueError("Whisper audio frontend expects rank 1 or 2 waveform input.")


def build_fixed_length_waveform(waveform, num_samples):
    waveform = ops.pad(waveform, [[0, 0], [0, num_samples]])
    return waveform[:, :num_samples]


def build_stft_waveform(waveform, num_fft_bins):
    pad_width = num_fft_bins // 2
    return ops.pad(
        waveform,
        [[0, 0], [pad_width, pad_width]],
        mode="reflect",
    )


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


def compute_log_mel_features(mel_spectrogram):
    minimum_value = ops.cast(1e-10, mel_spectrogram.dtype)
    mel_spectrogram = ops.maximum(mel_spectrogram, minimum_value)
    log_spectrogram = ops.log(mel_spectrogram)
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


def build_whisper_mel_filters(num_mels, num_fft_bins, sampling_rate, dtype):
    weights = np.zeros(
        (num_mels, int(1 + num_fft_bins // 2)),
        dtype=np.float32,
    )
    fft_frequencies = np.fft.rfftfreq(
        n=num_fft_bins,
        d=1.0 / sampling_rate,
    )
    min_mel = 0.0
    max_mel = 45.245640471924965
    mel_values = np.linspace(min_mel, max_mel, num_mels + 2)
    frequency_minimum = 0.0
    frequency_scale = 200.0 / 3
    mel_frequencies = frequency_minimum + frequency_scale * mel_values
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - frequency_minimum) / frequency_scale
    log_step = np.log(6.4) / 27.0
    is_log_region = mel_values >= min_log_mel
    mel_frequencies[is_log_region] = min_log_hz * np.exp(
        log_step * (mel_values[is_log_region] - min_log_mel)
    )
    frequency_differences = np.diff(mel_frequencies)
    ramps = np.subtract.outer(mel_frequencies, fft_frequencies)
    for mel_index in range(num_mels):
        lower = -ramps[mel_index] / frequency_differences[mel_index]
        upper = ramps[mel_index + 2] / frequency_differences[mel_index + 1]
        weights[mel_index] = np.maximum(0, np.minimum(lower, upper))
    energy_normalization = 2.0 / (
        mel_frequencies[2 : num_mels + 2] - mel_frequencies[:num_mels]
    )
    weights *= energy_normalization[:, np.newaxis]
    weights = np.transpose(weights)
    return np.asarray(weights, dtype=dtype)


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


def build_causal_attention_mask(batch_size, input_length, output_length):
    query_indices = ops.arange(output_length, dtype="float32")
    query_indices = ops.expand_dims(query_indices, axis=1)
    key_indices = ops.arange(input_length, dtype="float32")
    mask = ops.expand_dims(query_indices >= key_indices, axis=0)
    return ops.broadcast_to(mask, (batch_size, output_length, input_length))


def build_decoder_self_attention_mask(decoder_sequence, decoder_padding_mask):
    batch_size = ops.shape(decoder_sequence)[0]
    sequence_length = ops.shape(decoder_sequence)[1]
    causal_mask = build_causal_attention_mask(
        batch_size, sequence_length, sequence_length
    )
    return build_padding_attention_mask(decoder_padding_mask, causal_mask)


def _index_to_einsum_variable(index):
    return string.ascii_lowercase[index]


def _build_projection_equation(free_dims, bound_dims, output_dims):
    input_string = ""
    kernel_string = ""
    output_string = ""
    bias_axes = ""
    letter_offset = 0
    for index in range(free_dims):
        character = _index_to_einsum_variable(index + letter_offset)
        input_string += character
        output_string += character
    letter_offset += free_dims
    for index in range(bound_dims):
        character = _index_to_einsum_variable(index + letter_offset)
        input_string += character
        kernel_string += character
    letter_offset += bound_dims
    for index in range(output_dims):
        character = _index_to_einsum_variable(index + letter_offset)
        kernel_string += character
        output_string += character
        bias_axes += character
    equation = f"{input_string},{kernel_string}->{output_string}"
    return equation, bias_axes, len(output_string)


def _build_output_shape(output_rank, known_last_dims):
    return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)


class WhisperMelSpectrogram(keras.layers.Layer):
    def __init__(
        self,
        num_mels=80,
        num_fft_bins=400,
        sampling_rate=16000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mels = num_mels
        self.num_fft_bins = num_fft_bins
        self.sampling_rate = sampling_rate

    def build(self, inputs_shape):
        mel_filters = build_whisper_mel_filters(
            self.num_mels,
            self.num_fft_bins,
            self.sampling_rate,
            self.compute_dtype,
        )
        self.mel_filters = self.add_weight(
            name="mel_filters",
            shape=mel_filters.shape,
            initializer=keras.initializers.Constant(mel_filters),
            trainable=False,
        )
        self.built = True

    def call(self, inputs):
        return ops.matmul(inputs, ops.convert_to_tensor(self.mel_filters))

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.num_mels,)


class WhisperAudioFrontend(keras.layers.Layer):
    def __init__(
        self,
        num_mels=80,
        num_fft_bins=400,
        stride=160,
        sampling_rate=16000,
        max_audio_length=30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mels = num_mels
        self.num_fft_bins = num_fft_bins
        self.stride = stride
        self.sampling_rate = sampling_rate
        self.max_audio_length = max_audio_length
        self.num_samples = sampling_rate * max_audio_length
        self.mel_spectrogram = WhisperMelSpectrogram(
            num_mels=num_mels,
            num_fft_bins=num_fft_bins,
            sampling_rate=sampling_rate,
            dtype=self.dtype_policy,
            name="mel_spectrogram",
        )

    def call(self, waveform):
        waveform = ops.convert_to_tensor(waveform, dtype=self.compute_dtype)
        waveform, squeeze_batch_axis = build_batched_waveform(waveform)
        waveform = build_fixed_length_waveform(waveform, self.num_samples)
        waveform = build_stft_waveform(waveform, self.num_fft_bins)
        real_part, imaginary_part = compute_stft_components(
            waveform,
            self.num_fft_bins,
            self.stride,
        )
        power_spectrogram = compute_power_spectrogram(real_part, imaginary_part)
        mel_spectrogram = self.mel_spectrogram(power_spectrogram)
        features = compute_log_mel_features(mel_spectrogram)
        return squeeze_features_batch_axis(features, squeeze_batch_axis)

    def compute_output_shape(self, input_shape):
        num_frames = self.num_samples // self.stride
        if len(input_shape) == 1:
            return (num_frames, self.num_mels)
        return (input_shape[0], num_frames, self.num_mels)


class WhisperPositionEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, initializer="glorot_uniform", **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = int(sequence_length)
        self.initializer = keras.initializers.get(initializer)

    def build(self, inputs_shape):
        feature_size = inputs_shape[-1]
        self.position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.sequence_length, feature_size],
            initializer=self.initializer,
            trainable=True,
        )
        self.built = True

    def call(self, inputs, start_index=0, positions=None):
        shape = ops.shape(inputs)
        sequence_length = shape[-2]
        position_embeddings = ops.convert_to_tensor(self.position_embeddings)
        if positions is None:
            positions = ops.arange(start_index, start_index + sequence_length)
            position_embeddings = ops.take(position_embeddings, positions, axis=0)
        else:
            if len(ops.shape(positions)) == 1:
                positions = ops.expand_dims(positions, axis=0)
            position_embeddings = ops.take(position_embeddings, positions, axis=0)
        return ops.broadcast_to(position_embeddings, shape)

    def compute_output_shape(self, input_shape):
        return input_shape


class WhisperTokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(
        self,
        vocabulary_size,
        sequence_length,
        embedding_dim,
        embeddings_initializer,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.token_embedding = ReversibleEmbedding(
            vocabulary_size,
            embedding_dim,
            tie_weights=True,
            embeddings_initializer=keras.initializers.get(
                embeddings_initializer
            ),
            mask_zero=False,
            dtype=self.dtype_policy,
            name="token_embedding",
        )
        self.position_embedding = WhisperPositionEmbedding(
            sequence_length=sequence_length,
            initializer=keras.initializers.get(embeddings_initializer),
            dtype=self.dtype_policy,
            name="position_embedding",
        )

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        embedding_dim = self.token_embedding.output_dim
        self.token_embedding.build(input_shape)
        self.position_embedding.build(input_shape + (embedding_dim,))
        self.built = True

    def call(self, inputs, start_index=0, positions=None):
        embedded_tokens = self.token_embedding(inputs)
        embedded_positions = self.position_embedding(
            embedded_tokens, start_index=start_index, positions=positions
        )
        return embedded_tokens + embedded_positions

    def compute_output_shape(self, input_shape):
        return tuple(input_shape) + (self.token_embedding.output_dim,)


class WhisperAttention(MultiHeadAttention):
    def build(self, query_shape, value_shape, key_shape=None):
        key_shape = value_shape if key_shape is None else key_shape
        if value_shape[1:-1] != key_shape[1:-1]:
            raise ValueError(
                "All non-feature dims of value and key must match. "
                f"Received {value_shape} and {key_shape}."
            )
        query_rank = len(query_shape)
        value_rank = len(value_shape)
        key_rank = len(key_shape)
        equation, bias_axes, output_rank = _build_projection_equation(
            query_rank - 1, bound_dims=1, output_dims=2
        )
        self._query_dense = keras.layers.EinsumDense(
            equation,
            output_shape=_build_output_shape(
                output_rank - 1, [self._num_heads, self._key_dim]
            ),
            bias_axes=bias_axes if self._use_bias else None,
            name="query",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._query_dense.build(query_shape)

        equation, _, output_rank = _build_projection_equation(
            key_rank - 1, bound_dims=1, output_dims=2
        )
        self._key_dense = keras.layers.EinsumDense(
            equation,
            output_shape=_build_output_shape(
                output_rank - 1, [self._num_heads, self._key_dim]
            ),
            bias_axes=None,
            name="key",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._key_dense.build(key_shape)

        equation, bias_axes, output_rank = _build_projection_equation(
            value_rank - 1, bound_dims=1, output_dims=2
        )
        self._value_dense = keras.layers.EinsumDense(
            equation,
            output_shape=_build_output_shape(
                output_rank - 1, [self._num_heads, self._value_dim]
            ),
            bias_axes=bias_axes if self._use_bias else None,
            name="value",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._value_dense.build(value_shape)

        self._build_attention(output_rank)
        self._output_dense = self._make_output_dense(
            query_shape,
            self._get_common_kwargs_for_sublayer(),
            "attention_output",
        )
        output_dense_input_shape = list(
            self._query_dense.compute_output_shape(query_shape)
        )
        output_dense_input_shape[-1] = self._value_dim
        self._output_dense.build(tuple(output_dense_input_shape))
        self.built = True


class WhisperEncoderBlock(keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        intermediate_dim,
        dropout=0.0,
        layer_norm_epsilon=1e-5,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon

    def build(self, inputs_shape):
        hidden_dim = inputs_shape[-1]
        key_dim = int(hidden_dim // self.num_heads)
        self.self_attention_layer = WhisperAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout,
            kernel_initializer=whisper_kernel_initializer(),
            bias_initializer="zeros",
            dtype=self.dtype_policy,
            name="self_attention_layer",
        )
        self.self_attention_layer.build(inputs_shape, inputs_shape)
        self.self_attention_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attention_layer_norm",
        )
        self.self_attention_layer_norm.build(inputs_shape)
        self.self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="self_attention_dropout",
        )
        self.feedforward_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="feedforward_layer_norm",
        )
        self.feedforward_layer_norm.build(inputs_shape)
        self.feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=keras.activations.gelu,
            kernel_initializer=whisper_kernel_initializer(),
            bias_initializer="zeros",
            dtype=self.dtype_policy,
            name="feedforward_intermediate_dense",
        )
        self.feedforward_intermediate_dense.build(inputs_shape)
        self.feedforward_output_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=whisper_kernel_initializer(),
            bias_initializer="zeros",
            dtype=self.dtype_policy,
            name="feedforward_output_dense",
        )
        intermediate_shape = list(inputs_shape)
        intermediate_shape[-1] = self.intermediate_dim
        self.feedforward_output_dense.build(tuple(intermediate_shape))
        self.feedforward_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="feedforward_dropout",
        )
        self.built = True

    def call(self, inputs, training=None):
        residual = inputs
        hidden = self.self_attention_layer_norm(inputs)
        hidden = self.self_attention_layer(
            query=hidden,
            value=hidden,
            attention_mask=None,
            training=training,
        )
        hidden = self.self_attention_dropout(hidden, training=training)
        hidden = hidden + residual

        residual = hidden
        hidden = self.feedforward_layer_norm(hidden)
        hidden = self.feedforward_intermediate_dense(hidden)
        hidden = self.feedforward_output_dense(hidden)
        hidden = self.feedforward_dropout(hidden, training=training)
        return hidden + residual


class WhisperDecoderBlock(keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        intermediate_dim,
        dropout=0.0,
        layer_norm_epsilon=1e-5,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon

    def build(
        self,
        decoder_sequence_shape,
        encoder_sequence_shape,
        decoder_padding_mask_shape=None,
    ):
        decoder_shape = decoder_sequence_shape
        encoder_shape = encoder_sequence_shape
        hidden_dim = decoder_shape[-1]
        head_dim = int(hidden_dim // self.num_heads)

        self.self_attention = WhisperAttention(
            num_heads=self.num_heads,
            key_dim=head_dim,
            dropout=self.dropout,
            kernel_initializer=whisper_kernel_initializer(),
            bias_initializer="zeros",
            dtype=self.dtype_policy,
            name="self_attention",
        )
        self.self_attention.build(decoder_shape, decoder_shape)
        self.self_attention_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attention_layer_norm",
        )
        self.self_attention_layer_norm.build(decoder_shape)
        self.self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="self_attention_dropout",
        )

        self.cross_attention = WhisperAttention(
            num_heads=self.num_heads,
            key_dim=head_dim,
            value_dim=head_dim,
            dropout=self.dropout,
            kernel_initializer=whisper_kernel_initializer(),
            bias_initializer="zeros",
            dtype=self.dtype_policy,
            name="cross_attention",
        )
        self.cross_attention.build(decoder_shape, encoder_shape)
        self.cross_attention_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="cross_attention_layer_norm",
        )
        self.cross_attention_layer_norm.build(decoder_shape)
        self.cross_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="cross_attention_dropout",
        )

        self.feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=keras.activations.gelu,
            kernel_initializer=whisper_kernel_initializer(),
            bias_initializer="zeros",
            dtype=self.dtype_policy,
            name="feedforward_intermediate_dense",
        )
        self.feedforward_intermediate_dense.build(decoder_shape)
        self.feedforward_output_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=whisper_kernel_initializer(),
            bias_initializer="zeros",
            dtype=self.dtype_policy,
            name="feedforward_output_dense",
        )
        intermediate_shape = list(decoder_shape)
        intermediate_shape[-1] = self.intermediate_dim
        self.feedforward_output_dense.build(tuple(intermediate_shape))
        self.feedforward_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="feedforward_layer_norm",
        )
        self.feedforward_layer_norm.build(decoder_shape)
        self.feedforward_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="feedforward_dropout",
        )
        self.built = True

    def call(
        self,
        decoder_sequence,
        encoder_sequence,
        decoder_padding_mask,
        encoder_padding_mask=None,
        training=None,
    ):
        self_attention_mask = build_decoder_self_attention_mask(
            decoder_sequence, decoder_padding_mask
        )
        residual = decoder_sequence
        hidden = self.self_attention_layer_norm(decoder_sequence)
        hidden = self.self_attention(
            query=hidden,
            value=hidden,
            attention_mask=self_attention_mask,
            training=training,
        )
        hidden = self.self_attention_dropout(hidden, training=training)
        hidden = hidden + residual

        cross_attention_mask = build_padding_attention_mask(encoder_padding_mask)
        residual = hidden
        hidden = self.cross_attention_layer_norm(hidden)
        hidden = self.cross_attention(
            query=hidden,
            value=encoder_sequence,
            attention_mask=cross_attention_mask,
            training=training,
        )
        hidden = self.cross_attention_dropout(hidden, training=training)
        hidden = hidden + residual

        residual = hidden
        hidden = self.feedforward_layer_norm(hidden)
        hidden = self.feedforward_intermediate_dense(hidden)
        hidden = self.feedforward_output_dense(hidden)
        hidden = self.feedforward_dropout(hidden, training=training)
        return hidden + residual
