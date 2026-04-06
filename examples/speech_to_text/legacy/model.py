import keras
from keras import Model, ops
from keras.layers import Add, Conv1D, Dropout, Input, Lambda
from keras.layers import LayerNormalization, ReversibleEmbedding

from examples.speech_to_text.layers import build_attention_cache
from examples.speech_to_text.layers import build_mel_filters
from examples.speech_to_text.layers import cached_decoder_block
from examples.speech_to_text.layers import decoder_block
from examples.speech_to_text.layers import encoder_block
from examples.speech_to_text.layers import frontend
from examples.speech_to_text.layers import kernel_initializer
from examples.speech_to_text.layers import position_embedding


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


def Whisper(
    vocabulary_size,
    num_layers,
    num_heads,
    hidden_dim,
    intermediate_dim,
    num_mels=80,
    dropout=0.0,
    max_encoder_sequence_length=3000,
    max_decoder_sequence_length=448,
    dtype="float32",
    name="whisper",
    weights=None,
):
    encoder_features = Input(shape=(None, num_mels), dtype="float32", name="encoder_features")
    decoder_token_ids = Input(shape=(None,), dtype="int32", name="decoder_token_ids")
    decoder_padding_mask = Input(shape=(None,), dtype="int32", name="decoder_padding_mask")
    encoder_output = _apply_encoder(
        encoder_features,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout,
        max_encoder_sequence_length,
        dtype,
    )
    decoder_output, decoder_embeddings = _apply_decoder(
        decoder_token_ids,
        decoder_padding_mask,
        encoder_output,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout,
        max_decoder_sequence_length,
        dtype,
    )
    logits = decoder_embeddings(decoder_output, reverse=True)
    model = Model([encoder_features, decoder_token_ids, decoder_padding_mask], [encoder_output, decoder_output, logits], name=name)
    if weights is not None:
        from examples.speech_to_text.weights import load_preset_weights

        load_preset_weights(model, weights, dtype=dtype)
    return model


def WhisperEncoder(
    vocabulary_size,
    num_layers,
    num_heads,
    hidden_dim,
    intermediate_dim,
    num_mels=80,
    dropout=0.0,
    max_encoder_sequence_length=3000,
    max_decoder_sequence_length=448,
    dtype="float32",
    name="whisper_encoder",
    weights=None,
):
    encoder_features = Input(shape=(None, num_mels), dtype="float32", name="encoder_features")
    encoder_output = _apply_encoder(
        encoder_features,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout,
        max_encoder_sequence_length,
        dtype,
    )
    model = Model(encoder_features, encoder_output, name=name)
    if weights is not None:
        from examples.speech_to_text.weights import load_preset_weights

        load_preset_weights(model, weights, dtype=dtype, model_kind="encoder")
    return model


def WhisperDecoder(
    vocabulary_size,
    num_layers,
    num_heads,
    hidden_dim,
    intermediate_dim,
    num_mels=80,
    dropout=0.0,
    max_encoder_sequence_length=3000,
    max_decoder_sequence_length=448,
    dtype="float32",
    name="whisper_decoder",
    weights=None,
):
    decoder_token_ids = Input(shape=(None,), dtype="int32", name="decoder_token_ids")
    decoder_padding_mask = Input(shape=(None,), dtype="int32", name="decoder_padding_mask")
    encoder_output = Input(shape=(None, hidden_dim), dtype="float32", name="encoder_output")
    decoder_output, _ = _apply_decoder(
        decoder_token_ids,
        decoder_padding_mask,
        encoder_output,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout,
        max_decoder_sequence_length,
        dtype,
    )
    model = Model([decoder_token_ids, decoder_padding_mask, encoder_output], decoder_output, name=name)
    if weights is not None:
        from examples.speech_to_text.weights import load_preset_weights

        load_preset_weights(model, weights, dtype=dtype, model_kind="decoder")
    return model


def WhisperCrossCache(
    vocabulary_size,
    num_layers,
    num_heads,
    hidden_dim,
    intermediate_dim,
    num_mels=80,
    dropout=0.0,
    max_encoder_sequence_length=3000,
    max_decoder_sequence_length=448,
    dtype="float32",
    name="whisper_cross_cache",
    weights=None,
):
    del vocabulary_size, intermediate_dim, num_mels, dropout
    del max_encoder_sequence_length, max_decoder_sequence_length
    encoder_output = Input(shape=(None, hidden_dim), dtype="float32", name="encoder_output")
    key_dim = int(hidden_dim // num_heads)
    caches = []
    for layer_index in range(num_layers):
        cache = build_attention_cache(
            encoder_output,
            None,
            num_heads,
            key_dim,
            key_dim,
            True,
            kernel_initializer(),
            "zeros",
            dtype,
            "transformer_decoder_layer_{}_cross_attention".format(
                layer_index
            ),
        )
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
        from examples.speech_to_text.weights import load_preset_weights

        load_preset_weights(model, weights, dtype=dtype, model_kind="cross_cache")
    return model


def WhisperDecoderStep(
    vocabulary_size,
    num_layers,
    num_heads,
    hidden_dim,
    intermediate_dim,
    num_mels=80,
    dropout=0.0,
    max_encoder_sequence_length=3000,
    max_decoder_sequence_length=448,
    dtype="float32",
    name="whisper_decoder_step",
    weights=None,
):
    del num_mels, max_encoder_sequence_length
    key_dim = int(hidden_dim // num_heads)
    decoder_token_ids = Input(shape=(1,), dtype="int32", name="decoder_token_ids")
    self_attention_cache = Input(
        shape=(num_layers, 2, None, num_heads, key_dim),
        dtype="float32",
        name="self_attention_cache",
    )
    cross_attention_cache = Input(
        shape=(num_layers, 2, None, num_heads, key_dim),
        dtype="float32",
        name="cross_attention_cache",
    )
    cache_update_index = Input(shape=(), dtype="int32", name="cache_update_index")
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

    decoder_embeddings = ReversibleEmbedding(
        vocabulary_size,
        hidden_dim,
        tie_weights=True,
        embeddings_initializer=kernel_initializer(),
        mask_zero=False,
        dtype=dtype,
        name="decoder_token_embedding",
    )
    hidden = decoder_embeddings(decoder_token_ids)
    position_embeddings = position_embedding(
        hidden,
        max_decoder_sequence_length,
        kernel_initializer(),
        0,
        positions,
        True,
        dtype,
        "decoder_position_embedding",
    )
    hidden = Add(dtype=dtype, name="decoder_embeddings_add")(
        (hidden, position_embeddings)
    )
    hidden = Dropout(
        dropout, dtype=dtype, name="decoder_embeddings_dropout"
    )(hidden)
    updated_self_attention_caches = []
    for layer_index in range(num_layers):
        layer_self_attention_cache = Lambda(
            lambda x, index=layer_index: x[:, index, ...],
            output_shape=(2, None, num_heads, key_dim),
            name="transformer_decoder_layer_{}_self_attention_cache_slice".format(
                layer_index
            ),
        )(self_attention_cache)
        layer_cross_attention_cache = Lambda(
            lambda x, index=layer_index: x[:, index, ...],
            output_shape=(2, None, num_heads, key_dim),
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
            dtype,
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
    hidden = LayerNormalization(
        axis=-1,
        epsilon=1e-5,
        dtype=dtype,
        name="decoder_layer_norm",
    )(hidden)
    logits = decoder_embeddings(hidden, reverse=True)
    model = Model(
        [
            decoder_token_ids,
            self_attention_cache,
            cross_attention_cache,
            cache_update_index,
        ],
        [logits, updated_self_attention_cache],
        name=name,
    )
    if weights is not None:
        from examples.speech_to_text.weights import load_preset_weights

        load_preset_weights(
            model, weights, dtype=dtype, model_kind="decoder_step"
        )
    return model


def WhisperFrontend(
    num_mels=80,
    num_fft_bins=400,
    stride=160,
    sampling_rate=16000,
    max_audio_length=30,
    dtype="float32",
    name="whisper_frontend",
):
    waveform = Input(shape=(None,), dtype="float32", name="waveform")
    mel_filters = build_mel_filters(num_mels, num_fft_bins, sampling_rate, dtype)
    mel_filters = ops.convert_to_tensor(mel_filters, dtype=dtype)
    features = frontend(waveform, mel_filters, num_fft_bins, stride, sampling_rate, max_audio_length, dtype)
    return Model(waveform, features, name=name)


def WhisperTinyEn(dtype="float32"):
    return Whisper(dtype=dtype, name="whisper_tiny_en", weights="whisper_tiny_en", **CONFIGS["whisper_tiny_en"])


def WhisperBaseEn(dtype="float32"):
    return Whisper(dtype=dtype, name="whisper_base_en", weights="whisper_base_en", **CONFIGS["whisper_base_en"])


def WhisperSmallEn(dtype="float32"):
    return Whisper(dtype=dtype, name="whisper_small_en", weights="whisper_small_en", **CONFIGS["whisper_small_en"])


def WhisperMediumEn(dtype="float32"):
    return Whisper(dtype=dtype, name="whisper_medium_en", weights="whisper_medium_en", **CONFIGS["whisper_medium_en"])


def WhisperTinyMulti(dtype="float32"):
    return Whisper(dtype=dtype, name="whisper_tiny_multi", weights="whisper_tiny_multi", **CONFIGS["whisper_tiny_multi"])


def WhisperBaseMulti(dtype="float32"):
    return Whisper(dtype=dtype, name="whisper_base_multi", weights="whisper_base_multi", **CONFIGS["whisper_base_multi"])


def WhisperSmallMulti(dtype="float32"):
    return Whisper(dtype=dtype, name="whisper_small_multi", weights="whisper_small_multi", **CONFIGS["whisper_small_multi"])


def WhisperMediumMulti(dtype="float32"):
    return Whisper(dtype=dtype, name="whisper_medium_multi", weights="whisper_medium_multi", **CONFIGS["whisper_medium_multi"])


def WhisperLargeMulti(dtype="float32"):
    return Whisper(dtype=dtype, name="whisper_large_multi", weights="whisper_large_multi", **CONFIGS["whisper_large_multi"])


def WhisperLargeMultiV2(dtype="float32"):
    return Whisper(dtype=dtype, name="whisper_large_multi_v2", weights="whisper_large_multi_v2", **CONFIGS["whisper_large_multi_v2"])


def _apply_encoder(
    encoder_features,
    num_layers,
    num_heads,
    hidden_dim,
    intermediate_dim,
    dropout,
    max_encoder_sequence_length,
    dtype,
):
    hidden = Conv1D(filters=hidden_dim, kernel_size=3, strides=1, padding="same", dtype=dtype, name="encoder_token_embedding_conv_layer_1")(encoder_features)
    hidden = keras.activations.gelu(hidden, approximate=False)
    hidden = Lambda(lambda x: ops.pad(x, [[0, 0], [1, 1], [0, 0]]), output_shape=lambda shape: (shape[0], None, shape[2]), name="encoder_padder")(hidden)
    hidden = Conv1D(filters=hidden_dim, kernel_size=3, strides=2, padding="valid", dtype=dtype, name="encoder_token_embedding_conv_layer_2")(hidden)
    hidden = keras.activations.gelu(hidden, approximate=False)
    positions = position_embedding(hidden, max_encoder_sequence_length // 2, kernel_initializer(), 0, None, False, dtype, "encoder_position_embedding")
    hidden = Add(dtype=dtype, name="encoder_embeddings_add")((hidden, positions))
    hidden = Dropout(dropout, dtype=dtype, name="encoder_embeddings_dropout")(hidden)
    for layer_index in range(num_layers):
        hidden = encoder_block(
            hidden,
            num_heads,
            intermediate_dim,
            dropout,
            1e-5,
            dtype,
            "transformer_encoder_layer_{}".format(layer_index),
        )
    return LayerNormalization(axis=-1, epsilon=1e-5, dtype=dtype, name="encoder_layer_norm")(hidden)


def _apply_decoder(
    decoder_token_ids,
    decoder_padding_mask,
    encoder_output,
    vocabulary_size,
    num_layers,
    num_heads,
    hidden_dim,
    intermediate_dim,
    dropout,
    max_decoder_sequence_length,
    dtype,
):
    decoder_embeddings = ReversibleEmbedding(vocabulary_size, hidden_dim, tie_weights=True, embeddings_initializer=kernel_initializer(), mask_zero=False, dtype=dtype, name="decoder_token_embedding")
    hidden = decoder_embeddings(decoder_token_ids)
    positions = position_embedding(hidden, max_decoder_sequence_length, kernel_initializer(), 0, None, True, dtype, "decoder_position_embedding")
    hidden = Add(dtype=dtype, name="decoder_embeddings_add")((hidden, positions))
    hidden = Dropout(dropout, dtype=dtype, name="decoder_embeddings_dropout")(hidden)
    for layer_index in range(num_layers):
        hidden = decoder_block(
            hidden,
            encoder_output,
            decoder_padding_mask,
            None,
            num_heads,
            intermediate_dim,
            dropout,
            1e-5,
            dtype,
            "transformer_decoder_layer_{}".format(layer_index),
        )
    hidden = LayerNormalization(axis=-1, epsilon=1e-5, dtype=dtype, name="decoder_layer_norm")(hidden)
    return hidden, decoder_embeddings
