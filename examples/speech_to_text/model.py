import keras
from keras import Model
from keras import ops
from keras.layers import Input

from examples.speech_to_text.layers import decoder_block
from examples.speech_to_text.layers import encoder_block
from examples.speech_to_text.layers import frontend
from examples.speech_to_text.layers import kernel_initializer
from examples.speech_to_text.layers import position_embedding


# Field order:
# variant_name, vocabulary_size, num_layers, num_heads, hidden_dim,
# intermediate_dim, num_mels, max_encoder_sequence_length,
# max_decoder_sequence_length, is_multilingual
WHISPER_VARIANTS = (
    ("whisper_tiny_en", 51864, 4, 6, 384, 1536, 80, 3000, 448, False),
    ("whisper_base_en", 51864, 6, 8, 512, 2048, 80, 3000, 448, False),
    ("whisper_small_en", 51864, 12, 12, 768, 3072, 80, 3000, 448, False),
    ("whisper_medium_en", 51864, 24, 16, 1024, 4096, 80, 3000, 448, False),
    ("whisper_tiny_multi", 51865, 4, 6, 384, 1536, 80, 3000, 448, True),
    ("whisper_base_multi", 51865, 6, 8, 512, 2048, 80, 3000, 448, True),
    ("whisper_small_multi", 51865, 12, 12, 768, 3072, 80, 3000, 448, True),
    ("whisper_medium_multi", 51865, 24, 16, 1024, 4096, 80, 3000, 448, True),
    ("whisper_large_multi", 51865, 32, 20, 1280, 5120, 80, 3000, 448, True),
    (
        "whisper_large_multi_v2",
        51865,
        32,
        20,
        1280,
        5120,
        80,
        3000,
        448,
        True,
    ),
)


def get_whisper_variant_names():
    return tuple(variant_values[0] for variant_values in WHISPER_VARIANTS)


def find_whisper_variant_values(variant_name):
    for variant_values in WHISPER_VARIANTS:
        if variant_values[0] == variant_name:
            return variant_values
    raise ValueError("Unknown Whisper variant: {}".format(variant_name))


def build_whisper_variant_model(
    variant_name, dtype="float32", name=None
):
    variant_values = find_whisper_variant_values(variant_name)
    if name is None:
        name = variant_name
    (
        _,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        num_mels,
        max_encoder_sequence_length,
        max_decoder_sequence_length,
        _,
    ) = variant_values
    return build_whisper_core_model(
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        num_mels,
        0.0,
        max_encoder_sequence_length,
        max_decoder_sequence_length,
        dtype,
        name,
    )


def build_whisper_variant_logits_model(
    variant_name, dtype="float32", name=None
):
    variant_values = find_whisper_variant_values(variant_name)
    if name is None:
        name = "{}_logits".format(variant_name)
    (
        _,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        num_mels,
        max_encoder_sequence_length,
        max_decoder_sequence_length,
        _,
    ) = variant_values
    return build_whisper_core_logits_model(
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        num_mels,
        0.0,
        max_encoder_sequence_length,
        max_decoder_sequence_length,
        dtype,
        name,
    )


def build_whisper_audio_frontend(
    num_mels=80,
    num_fft_bins=400,
    stride=160,
    sampling_rate=16000,
    max_audio_length=30,
    dtype="float32",
    name="whisper_audio_frontend",
):
    waveform = Input(
        shape=(None,),
        dtype="float32",
        name="waveform",
    )
    features = apply_whisper_audio_frontend(
        waveform,
        num_mels,
        num_fft_bins,
        stride,
        sampling_rate,
        max_audio_length,
        dtype,
    )
    return Model(waveform, features, name=name)


def build_whisper_waveform_to_features_model(
    num_mels=80,
    num_fft_bins=400,
    stride=160,
    sampling_rate=16000,
    max_audio_length=30,
    dtype="float32",
    name="whisper_waveform_to_features",
):
    return build_whisper_audio_frontend(
        num_mels,
        num_fft_bins,
        stride,
        sampling_rate,
        max_audio_length,
        dtype,
        name,
    )


def build_whisper_encoder_model(
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
    encoder_output = apply_whisper_encoder(
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


def build_whisper_decoder_model(
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
    encoder_sequence_output = Input(
        shape=(None, hidden_dim),
        dtype=dtype,
        name="encoder_sequence_output",
    )
    decoder_output, _ = apply_whisper_decoder(
        decoder_token_ids,
        decoder_padding_mask,
        encoder_sequence_output,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout,
        max_decoder_sequence_length,
        dtype,
    )
    inputs = [decoder_token_ids, decoder_padding_mask, encoder_sequence_output]
    return Model(inputs, decoder_output, name=name)


def build_whisper_core_model(
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
    name="whisper_core",
):
    encoder_features, decoder_token_ids, decoder_padding_mask = build_model_inputs(
        num_mels
    )
    encoder_output = apply_whisper_encoder(
        encoder_features,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout,
        max_encoder_sequence_length,
        dtype,
    )
    decoder_output, _ = apply_whisper_decoder(
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
    inputs = [encoder_features, decoder_token_ids, decoder_padding_mask]
    outputs = [encoder_output, decoder_output]
    return Model(inputs, outputs, name=name)


def build_whisper_core_logits_model(
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
    name="whisper_core_logits",
):
    encoder_features, decoder_token_ids, decoder_padding_mask = build_model_inputs(
        num_mels
    )
    encoder_output = apply_whisper_encoder(
        encoder_features,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout,
        max_encoder_sequence_length,
        dtype,
    )
    decoder_output, decoder_embeddings = apply_whisper_decoder(
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
    inputs = [encoder_features, decoder_token_ids, decoder_padding_mask]
    outputs = [encoder_output, decoder_output, logits]
    return Model(inputs, outputs, name=name)


def build_whisper_base_en_model(dtype="float32", name="whisper_base_en"):
    return build_whisper_variant_model("whisper_base_en", dtype, name)


def build_whisper_base_en_encoder_model(
    dtype="float32", name="whisper_base_en_encoder"
):
    return build_whisper_encoder_model(6, 8, 512, 2048, 80, 0.0, 3000, dtype, name)


def build_whisper_base_en_decoder_model(
    dtype="float32", name="whisper_base_en_decoder"
):
    return build_whisper_decoder_model(
        51864, 6, 8, 512, 2048, 0.0, 448, dtype, name
    )


def build_whisper_tiny_en_model(dtype="float32", name="whisper_tiny_en"):
    return build_whisper_variant_model("whisper_tiny_en", dtype, name)


def build_whisper_small_en_model(dtype="float32", name="whisper_small_en"):
    return build_whisper_variant_model("whisper_small_en", dtype, name)


def build_whisper_medium_en_model(
    dtype="float32", name="whisper_medium_en"
):
    return build_whisper_variant_model("whisper_medium_en", dtype, name)


def build_whisper_base_en_logits_model(
    dtype="float32", name="whisper_base_en_logits"
):
    return build_whisper_variant_logits_model("whisper_base_en", dtype, name)


def build_whisper_tiny_en_logits_model(
    dtype="float32", name="whisper_tiny_en_logits"
):
    return build_whisper_variant_logits_model("whisper_tiny_en", dtype, name)


def build_whisper_small_en_logits_model(
    dtype="float32", name="whisper_small_en_logits"
):
    return build_whisper_variant_logits_model("whisper_small_en", dtype, name)


def build_whisper_medium_en_logits_model(
    dtype="float32", name="whisper_medium_en_logits"
):
    return build_whisper_variant_logits_model("whisper_medium_en", dtype, name)


def build_whisper_tiny_multi_model(dtype="float32", name="whisper_tiny_multi"):
    return build_whisper_variant_model("whisper_tiny_multi", dtype, name)


def build_whisper_base_multi_model(dtype="float32", name="whisper_base_multi"):
    return build_whisper_variant_model("whisper_base_multi", dtype, name)


def build_whisper_small_multi_model(
    dtype="float32", name="whisper_small_multi"
):
    return build_whisper_variant_model("whisper_small_multi", dtype, name)


def build_whisper_medium_multi_model(
    dtype="float32", name="whisper_medium_multi"
):
    return build_whisper_variant_model("whisper_medium_multi", dtype, name)


def build_whisper_large_multi_model(
    dtype="float32", name="whisper_large_multi"
):
    return build_whisper_variant_model("whisper_large_multi", dtype, name)


def build_whisper_large_multi_v2_model(
    dtype="float32", name="whisper_large_multi_v2"
):
    return build_whisper_variant_model("whisper_large_multi_v2", dtype, name)


def build_whisper_tiny_multi_logits_model(
    dtype="float32", name="whisper_tiny_multi_logits"
):
    return build_whisper_variant_logits_model("whisper_tiny_multi", dtype, name)


def build_whisper_base_multi_logits_model(
    dtype="float32", name="whisper_base_multi_logits"
):
    return build_whisper_variant_logits_model("whisper_base_multi", dtype, name)


def build_whisper_small_multi_logits_model(
    dtype="float32", name="whisper_small_multi_logits"
):
    return build_whisper_variant_logits_model("whisper_small_multi", dtype, name)


def build_whisper_medium_multi_logits_model(
    dtype="float32", name="whisper_medium_multi_logits"
):
    return build_whisper_variant_logits_model("whisper_medium_multi", dtype, name)


def build_whisper_large_multi_logits_model(
    dtype="float32", name="whisper_large_multi_logits"
):
    return build_whisper_variant_logits_model("whisper_large_multi", dtype, name)


def build_whisper_large_multi_v2_logits_model(
    dtype="float32", name="whisper_large_multi_v2_logits"
):
    return build_whisper_variant_logits_model(
        "whisper_large_multi_v2", dtype, name
    )


def build_whisper_base_en_waveform_to_features_model(
    dtype="float32",
    name="whisper_base_en_waveform_to_features",
):
    return build_whisper_waveform_to_features_model(
        80,
        400,
        160,
        16000,
        30,
        dtype,
        name,
    )


def build_model_inputs(num_mels):
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


def apply_whisper_audio_frontend(
    waveform,
    num_mels,
    num_fft_bins,
    stride,
    sampling_rate,
    max_audio_length,
    dtype,
):
    return frontend(
        waveform,
        num_mels,
        num_fft_bins,
        stride,
        sampling_rate,
        max_audio_length,
        dtype,
    )


def apply_whisper_encoder(
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
    hidden = build_encoder_position_embeddings(
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
            f"transformer_encoder_layer_{layer_index}",
        )
    hidden = keras.layers.LayerNormalization(
        axis=-1,
        epsilon=1e-5,
        dtype=dtype,
        name="encoder_layer_norm",
    )(hidden)
    return hidden


def apply_whisper_decoder(
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
            f"transformer_decoder_layer_{layer_index}",
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


def build_encoder_position_embeddings(
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
