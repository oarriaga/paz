import keras
from keras import Model
from keras import ops
from keras.layers import Input

from examples.speech_to_text.layers import build_mel_filters
from examples.speech_to_text.layers import decoder_block
from examples.speech_to_text.layers import encoder_block
from examples.speech_to_text.layers import frontend
from examples.speech_to_text.layers import kernel_initializer
from examples.speech_to_text.layers import position_embedding


WHISPER_CONFIGS = {
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

WHISPER_VARIANTS = tuple(
    (
        variant_name,
        config["vocabulary_size"],
        config["num_layers"],
        config["num_heads"],
        config["hidden_dim"],
        config["intermediate_dim"],
        config["num_mels"],
        config["max_encoder_sequence_length"],
        config["max_decoder_sequence_length"],
        config["vocabulary_size"] == 51865,
    )
    for variant_name, config in WHISPER_CONFIGS.items()
)


def get_whisper_variant_names():
    return tuple(WHISPER_CONFIGS.keys())


def find_whisper_variant_config(variant_name):
    if variant_name not in WHISPER_CONFIGS:
        raise ValueError("Unknown Whisper variant: {}".format(variant_name))
    return dict(WHISPER_CONFIGS[variant_name])


def find_whisper_variant_values(variant_name):
    config = find_whisper_variant_config(variant_name)
    return (
        variant_name,
        config["vocabulary_size"],
        config["num_layers"],
        config["num_heads"],
        config["hidden_dim"],
        config["intermediate_dim"],
        config["num_mels"],
        config["max_encoder_sequence_length"],
        config["max_decoder_sequence_length"],
        config["vocabulary_size"] == 51865,
    )


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
    preset_name=None,
):
    encoder_features, decoder_token_ids, decoder_padding_mask = build_inputs(
        num_mels
    )
    encoder_output = apply_encoder(
        encoder_features,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout,
        max_encoder_sequence_length,
        dtype,
    )
    decoder_output, decoder_embeddings = apply_decoder(
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
    model = Model(
        [encoder_features, decoder_token_ids, decoder_padding_mask],
        [encoder_output, decoder_output, logits],
        name=name,
    )
    if preset_name is not None:
        from examples.speech_to_text.weights import load_preset_weights

        load_preset_weights(model, preset_name, "full", dtype=dtype)
    return model


def WhisperEncoder(
    num_layers,
    num_heads,
    hidden_dim,
    intermediate_dim,
    num_mels=80,
    dropout=0.0,
    max_encoder_sequence_length=3000,
    dtype="float32",
    name="whisper_encoder",
):
    encoder_features = Input(
        shape=(None, num_mels),
        dtype="float32",
        name="encoder_features",
    )
    encoder_output = apply_encoder(
        encoder_features,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout,
        max_encoder_sequence_length,
        dtype,
    )
    return Model(encoder_features, encoder_output, name=name)


def WhisperDecoder(
    vocabulary_size,
    num_layers,
    num_heads,
    hidden_dim,
    intermediate_dim,
    dropout=0.0,
    max_decoder_sequence_length=448,
    dtype="float32",
    name="whisper_decoder",
):
    decoder_token_ids = Input(
        shape=(None,),
        dtype="int32",
        name="decoder_token_ids",
    )
    decoder_padding_mask = Input(
        shape=(None,),
        dtype="int32",
        name="decoder_padding_mask",
    )
    encoder_output = Input(
        shape=(None, hidden_dim),
        dtype=dtype,
        name="encoder_sequence_output",
    )
    decoder_output, _ = apply_decoder(
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
    return Model(
        [decoder_token_ids, decoder_padding_mask, encoder_output],
        decoder_output,
        name=name,
    )


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
    mel_filters = build_mel_filters(
        num_mels,
        num_fft_bins,
        sampling_rate,
        dtype,
    )
    mel_filters = ops.convert_to_tensor(mel_filters, dtype=dtype)
    features = frontend(
        waveform,
        mel_filters,
        num_fft_bins,
        stride,
        sampling_rate,
        max_audio_length,
        dtype,
    )
    return Model(waveform, features, name=name)


def WhisperTinyEn(dtype="float32", name="whisper_tiny_en"):
    return build_variant("whisper_tiny_en", dtype, name)


def WhisperBaseEn(dtype="float32", name="whisper_base_en"):
    return build_variant("whisper_base_en", dtype, name)


def WhisperSmallEn(dtype="float32", name="whisper_small_en"):
    return build_variant("whisper_small_en", dtype, name)


def WhisperMediumEn(dtype="float32", name="whisper_medium_en"):
    return build_variant("whisper_medium_en", dtype, name)


def WhisperTinyMulti(dtype="float32", name="whisper_tiny_multi"):
    return build_variant("whisper_tiny_multi", dtype, name)


def WhisperBaseMulti(dtype="float32", name="whisper_base_multi"):
    return build_variant("whisper_base_multi", dtype, name)


def WhisperSmallMulti(dtype="float32", name="whisper_small_multi"):
    return build_variant("whisper_small_multi", dtype, name)


def WhisperMediumMulti(dtype="float32", name="whisper_medium_multi"):
    return build_variant("whisper_medium_multi", dtype, name)


def WhisperLargeMulti(dtype="float32", name="whisper_large_multi"):
    return build_variant("whisper_large_multi", dtype, name)


def WhisperLargeMultiV2(dtype="float32", name="whisper_large_multi_v2"):
    return build_variant("whisper_large_multi_v2", dtype, name)


def build_variant(variant_name, dtype, name):
    config = find_whisper_variant_config(variant_name)
    return Whisper(
        dtype=dtype,
        name=name,
        preset_name=variant_name,
        **config,
    )


def build_inputs(num_mels):
    encoder_features = Input(
        shape=(None, num_mels),
        dtype="float32",
        name="encoder_features",
    )
    decoder_token_ids = Input(
        shape=(None,),
        dtype="int32",
        name="decoder_token_ids",
    )
    decoder_padding_mask = Input(
        shape=(None,),
        dtype="int32",
        name="decoder_padding_mask",
    )
    return encoder_features, decoder_token_ids, decoder_padding_mask


def apply_encoder(
    encoder_features,
    num_layers,
    num_heads,
    hidden_dim,
    intermediate_dim,
    dropout,
    max_encoder_sequence_length,
    dtype,
):
    hidden = build_encoder_token_embedding(encoder_features, hidden_dim, dtype)
    hidden = build_encoder_position_embedding(
        hidden, dropout, max_encoder_sequence_length, dtype
    )
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
    return keras.layers.LayerNormalization(
        axis=-1,
        epsilon=1e-5,
        dtype=dtype,
        name="encoder_layer_norm",
    )(hidden)


def apply_decoder(
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
    decoder_embeddings = keras.layers.ReversibleEmbedding(
        vocabulary_size,
        hidden_dim,
        tie_weights=True,
        embeddings_initializer=kernel_initializer(),
        mask_zero=False,
        dtype=dtype,
        name="decoder_token_embedding",
    )
    hidden = decoder_embeddings(decoder_token_ids)
    positions = position_embedding(
        hidden,
        max_decoder_sequence_length,
        kernel_initializer(),
        0,
        None,
        True,
        dtype,
        "decoder_position_embedding",
    )
    hidden = keras.layers.Add(
        dtype=dtype,
        name="decoder_embeddings_add",
    )((hidden, positions))
    hidden = keras.layers.Dropout(
        dropout,
        dtype=dtype,
        name="decoder_embeddings_dropout",
    )(hidden)
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
    hidden = keras.layers.LayerNormalization(
        axis=-1,
        epsilon=1e-5,
        dtype=dtype,
        name="decoder_layer_norm",
    )(hidden)
    return hidden, decoder_embeddings


def build_encoder_token_embedding(encoder_features, hidden_dim, dtype):
    hidden = keras.layers.Conv1D(
        filters=hidden_dim,
        kernel_size=3,
        strides=1,
        padding="same",
        dtype=dtype,
        name="encoder_token_embedding_conv_layer_1",
    )(encoder_features)
    hidden = keras.activations.gelu(hidden, approximate=False)
    hidden = keras.layers.Lambda(
        lambda x: ops.pad(x, [[0, 0], [1, 1], [0, 0]]),
        output_shape=lambda shape: (shape[0], None, shape[2]),
        name="encoder_padder",
    )(hidden)
    hidden = keras.layers.Conv1D(
        filters=hidden_dim,
        kernel_size=3,
        strides=2,
        padding="valid",
        dtype=dtype,
        name="encoder_token_embedding_conv_layer_2",
    )(hidden)
    return keras.activations.gelu(hidden, approximate=False)


def build_encoder_position_embedding(
    hidden, dropout, max_encoder_sequence_length, dtype
):
    positions = position_embedding(
        hidden,
        max_encoder_sequence_length // 2,
        kernel_initializer(),
        0,
        None,
        False,
        dtype,
        "encoder_position_embedding",
    )
    hidden = keras.layers.Add(
        dtype=dtype,
        name="encoder_embeddings_add",
    )((hidden, positions))
    return keras.layers.Dropout(
        dropout,
        dtype=dtype,
        name="encoder_embeddings_dropout",
    )(hidden)
