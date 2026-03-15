import keras
from keras import Model, ops
from keras.layers import Add, Conv1D, Dropout, Input, Lambda
from keras.layers import LayerNormalization, ReversibleEmbedding

from examples.speech_to_text.layers import build_mel_filters
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
