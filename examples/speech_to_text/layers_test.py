import keras
import numpy as np
from keras import ops
from keras.layers import Input

from examples.speech_to_text.layers import attention
from examples.speech_to_text.layers import build_decoder_self_attention_mask
from examples.speech_to_text.layers import build_fixed_length_waveform
from examples.speech_to_text.layers import build_mel_filters
from examples.speech_to_text.layers import decoder_block
from examples.speech_to_text.layers import encoder_block
from examples.speech_to_text.layers import frontend
from examples.speech_to_text.layers import kernel_initializer
from examples.speech_to_text.layers import position_embedding
from examples.speech_to_text.layers import token_and_position_embedding
from examples.speech_to_text.layers_old import WhisperAttention
from examples.speech_to_text.layers_old import WhisperAudioFrontend
from examples.speech_to_text.layers_old import WhisperDecoderBlock
from examples.speech_to_text.layers_old import WhisperEncoderBlock
from examples.speech_to_text.layers_old import WhisperPositionEmbedding
from examples.speech_to_text.layers_old import WhisperTokenAndPositionEmbedding
from examples.speech_to_text.layers_old import build_decoder_self_attention_mask as build_decoder_self_attention_mask_old
from examples.speech_to_text.layers_old import build_fixed_length_waveform as build_fixed_length_waveform_old
from examples.speech_to_text.layers_old import build_whisper_mel_filters
from examples.speech_to_text.layers_old import whisper_kernel_initializer


WRAPPER_MODEL_NAMES = {
    "position_model",
    "position_with_positions_model",
    "token_model",
    "attention_model",
    "encoder_model",
    "decoder_model",
}


def build_weight_suffix(path):
    parts = path.split("/")
    if parts and parts[0] in WRAPPER_MODEL_NAMES:
        return "/".join(parts[1:])
    return path


def normalize_reference_suffix(path):
    suffix = build_weight_suffix(path)
    suffix = suffix.replace(
        "decoder_token_and_position_embedding/token_embedding/",
        "token_embedding/",
    )
    suffix = suffix.replace(
        "decoder_token_and_position_embedding/position_embedding/",
        "position_embedding/",
    )
    suffix = suffix.replace("/query/", "_query/")
    suffix = suffix.replace("/key/", "_key/")
    suffix = suffix.replace("/value/", "_value/")
    suffix = suffix.replace("/attention_output/", "_attention_output/")
    suffix = suffix.replace(
        "/self_attention_layer_query/",
        "_self_attention_layer_query/",
    )
    suffix = suffix.replace(
        "/self_attention_layer_key/",
        "_self_attention_layer_key/",
    )
    suffix = suffix.replace(
        "/self_attention_layer_value/",
        "_self_attention_layer_value/",
    )
    suffix = suffix.replace(
        "/self_attention_layer_attention_output/",
        "_self_attention_layer_attention_output/",
    )
    suffix = suffix.replace(
        "/self_attention_query/",
        "_self_attention_query/",
    )
    suffix = suffix.replace(
        "/self_attention_key/",
        "_self_attention_key/",
    )
    suffix = suffix.replace(
        "/self_attention_value/",
        "_self_attention_value/",
    )
    suffix = suffix.replace(
        "/self_attention_attention_output/",
        "_self_attention_attention_output/",
    )
    suffix = suffix.replace(
        "/cross_attention_query/",
        "_cross_attention_query/",
    )
    suffix = suffix.replace(
        "/cross_attention_key/",
        "_cross_attention_key/",
    )
    suffix = suffix.replace(
        "/cross_attention_value/",
        "_cross_attention_value/",
    )
    suffix = suffix.replace(
        "/cross_attention_attention_output/",
        "_cross_attention_attention_output/",
    )
    suffix = suffix.replace("/self_attention_layer_norm/", "_self_attention_layer_norm/")
    suffix = suffix.replace("/cross_attention_layer_norm/", "_cross_attention_layer_norm/")
    suffix = suffix.replace("/feedforward_layer_norm/", "_feedforward_layer_norm/")
    suffix = suffix.replace(
        "/feedforward_intermediate_dense/",
        "_feedforward_intermediate_dense/",
    )
    suffix = suffix.replace(
        "/feedforward_output_dense/",
        "_feedforward_output_dense/",
    )
    return suffix


def copy_reference_weights(reference_layer, target_model):
    weight_values = {}
    for weight in reference_layer.weights:
        suffix = normalize_reference_suffix(weight.path)
        weight_values[suffix] = ops.convert_to_numpy(weight)
    for weight in target_model.weights:
        suffix = build_weight_suffix(weight.path)
        weight.assign(weight_values[suffix])


def build_position_model(trainable=False):
    hidden = Input(shape=(3, 4), dtype="float32", name="hidden")
    outputs = position_embedding(
        hidden,
        3,
        kernel_initializer(),
        0,
        None,
        trainable,
        "float32",
        "encoder_position_embedding",
    )
    return keras.Model(hidden, outputs, name="position_model")


def build_position_with_positions_model(trainable=False):
    hidden = Input(shape=(3, 4), dtype="float32", name="hidden")
    positions = Input(shape=(3,), dtype="int32", name="positions")
    outputs = position_embedding(
        hidden,
        3,
        kernel_initializer(),
        0,
        positions,
        trainable,
        "float32",
        "encoder_position_embedding",
    )
    return keras.Model(
        [hidden, positions],
        outputs,
        name="position_with_positions_model",
    )


def build_token_and_position_model():
    token_ids = Input(shape=(3,), dtype="int32", name="token_ids")
    outputs = token_and_position_embedding(
        token_ids,
        10,
        6,
        4,
        kernel_initializer(),
        0,
        None,
        "float32",
    )
    return keras.Model(token_ids, outputs, name="token_model")


def build_attention_model(name):
    query = Input(shape=(2, 4), dtype="float32", name="query")
    value = Input(shape=(2, 4), dtype="float32", name="value")
    outputs = attention(
        query,
        value,
        None,
        None,
        2,
        2,
        None,
        0.0,
        True,
        False,
        kernel_initializer(),
        "zeros",
        "float32",
        name,
    )
    return keras.Model([query, value], outputs, name="attention_model")


def build_encoder_block_model():
    hidden = Input(shape=(2, 4), dtype="float32", name="hidden")
    outputs = encoder_block(
        hidden,
        2,
        8,
        0.0,
        1e-5,
        "float32",
        "transformer_encoder_layer_0",
    )
    return keras.Model(hidden, outputs, name="encoder_model")


def build_decoder_block_model():
    decoder_hidden = Input(shape=(2, 4), dtype="float32", name="decoder_hidden")
    encoder_hidden = Input(shape=(2, 4), dtype="float32", name="encoder_hidden")
    decoder_padding_mask = Input(shape=(2,), dtype="int32", name="mask")
    outputs = decoder_block(
        decoder_hidden,
        encoder_hidden,
        decoder_padding_mask,
        None,
        2,
        8,
        0.0,
        1e-5,
        "float32",
        "transformer_decoder_layer_0",
    )
    return keras.Model(
        [decoder_hidden, encoder_hidden, decoder_padding_mask],
        outputs,
        name="decoder_model",
    )


def test_fixed_length_waveform_matches_original():
    waveform = ops.convert_to_tensor([[1.0, 2.0]], dtype="float32")
    new_waveform = build_fixed_length_waveform(waveform, 5)
    old_waveform = build_fixed_length_waveform_old(waveform, 5)
    np.testing.assert_array_equal(
        ops.convert_to_numpy(new_waveform),
        ops.convert_to_numpy(old_waveform),
    )


def test_mel_filters_match_original():
    new_filters = build_mel_filters(80, 400, 16000, "float32")
    old_filters = build_whisper_mel_filters(80, 400, 16000, "float32")
    np.testing.assert_allclose(new_filters, old_filters, rtol=1e-7, atol=1e-7)


def test_frontend_matches_original_single_waveform():
    old_layer = WhisperAudioFrontend(
        max_audio_length=1,
        sampling_rate=16000,
        dtype="float32",
    )
    waveform = ops.linspace(
        ops.cast(-1.0, "float32"),
        ops.cast(1.0, "float32"),
        1600,
    )
    new_output = ops.convert_to_numpy(frontend(waveform, max_audio_length=1))
    old_output = ops.convert_to_numpy(old_layer(waveform))
    np.testing.assert_allclose(new_output, old_output, rtol=1e-5, atol=2e-5)


def test_frontend_matches_original_batch():
    old_layer = WhisperAudioFrontend(
        max_audio_length=1,
        sampling_rate=16000,
        dtype="float32",
    )
    waveform = ops.linspace(
        ops.cast(-1.0, "float32"),
        ops.cast(1.0, "float32"),
        1600,
    )
    waveform = ops.stack([waveform, waveform * 0.5], axis=0)
    new_output = ops.convert_to_numpy(frontend(waveform, max_audio_length=1))
    old_output = ops.convert_to_numpy(old_layer(waveform))
    np.testing.assert_allclose(new_output, old_output, rtol=1e-5, atol=2e-5)


def test_position_embedding_matches_original():
    hidden = ops.ones((2, 3, 4), dtype="float32")
    old_layer = WhisperPositionEmbedding(
        sequence_length=3,
        initializer=whisper_kernel_initializer(),
        trainable=False,
        name="encoder_position_embedding",
    )
    _ = old_layer(hidden)
    new_model = build_position_model(trainable=False)
    copy_reference_weights(old_layer, new_model)
    new_output = ops.convert_to_numpy(new_model(hidden))
    old_output = ops.convert_to_numpy(old_layer(hidden))
    np.testing.assert_allclose(new_output, old_output, rtol=1e-5, atol=1e-5)


def test_position_embedding_with_positions_matches_original():
    hidden = ops.ones((2, 3, 4), dtype="float32")
    positions = ops.convert_to_tensor([[0, 1, 2], [2, 1, 0]], dtype="int32")
    old_layer = WhisperPositionEmbedding(
        sequence_length=3,
        initializer=whisper_kernel_initializer(),
        trainable=False,
        name="encoder_position_embedding",
    )
    _ = old_layer(hidden)
    new_model = build_position_with_positions_model(trainable=False)
    copy_reference_weights(old_layer, new_model)
    new_output = ops.convert_to_numpy(new_model([hidden, positions]))
    old_output = ops.convert_to_numpy(old_layer(hidden, positions=positions))
    np.testing.assert_allclose(new_output, old_output, rtol=1e-5, atol=1e-5)


def test_token_and_position_embedding_matches_original():
    token_ids = ops.convert_to_tensor([[1, 2, 3]], dtype="int32")
    old_layer = WhisperTokenAndPositionEmbedding(
        10,
        6,
        4,
        whisper_kernel_initializer(),
        name="decoder_token_and_position_embedding",
    )
    _ = old_layer(token_ids)
    new_model = build_token_and_position_model()
    copy_reference_weights(old_layer, new_model)
    new_output = ops.convert_to_numpy(new_model(token_ids))
    old_output = ops.convert_to_numpy(old_layer(token_ids))
    np.testing.assert_allclose(new_output, old_output, rtol=1e-5, atol=1e-5)


def test_decoder_mask_matches_original():
    decoder_sequence = ops.ones((1, 4, 4), dtype="float32")
    decoder_padding_mask = ops.convert_to_tensor([[1, 1, 1, 0]], dtype="int32")
    new_mask = ops.convert_to_numpy(
        build_decoder_self_attention_mask(decoder_sequence, decoder_padding_mask)
    )
    old_mask = ops.convert_to_numpy(
        build_decoder_self_attention_mask_old(
            decoder_sequence,
            decoder_padding_mask,
        )
    )
    np.testing.assert_array_equal(new_mask, old_mask)


def test_self_attention_matches_original():
    hidden = ops.convert_to_tensor(
        [[[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]]], dtype="float32"
    )
    old_layer = WhisperAttention(
        num_heads=2,
        key_dim=2,
        dropout=0.0,
        kernel_initializer=whisper_kernel_initializer(),
        bias_initializer="zeros",
        name="self_attention_layer",
    )
    old_layer.build(hidden.shape, hidden.shape)
    new_model = build_attention_model("self_attention_layer")
    copy_reference_weights(old_layer, new_model)
    new_output = ops.convert_to_numpy(new_model([hidden, hidden]))
    old_output = ops.convert_to_numpy(old_layer(hidden, hidden))
    np.testing.assert_allclose(new_output, old_output, rtol=1e-5, atol=1e-5)


def test_cross_attention_matches_original():
    decoder_hidden = ops.convert_to_tensor(
        [[[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]]], dtype="float32"
    )
    encoder_hidden = ops.convert_to_tensor(
        [[[4.0, 3.0, 2.0, 1.0], [3.0, 2.0, 1.0, 0.0]]], dtype="float32"
    )
    old_layer = WhisperAttention(
        num_heads=2,
        key_dim=2,
        value_dim=2,
        dropout=0.0,
        kernel_initializer=whisper_kernel_initializer(),
        bias_initializer="zeros",
        name="cross_attention",
    )
    old_layer.build(decoder_hidden.shape, encoder_hidden.shape)
    new_model = build_attention_model("cross_attention")
    copy_reference_weights(old_layer, new_model)
    new_output = ops.convert_to_numpy(
        new_model([decoder_hidden, encoder_hidden])
    )
    old_output = ops.convert_to_numpy(
        old_layer(decoder_hidden, encoder_hidden)
    )
    np.testing.assert_allclose(new_output, old_output, rtol=1e-5, atol=1e-5)


def test_encoder_block_matches_original():
    hidden = ops.convert_to_tensor(
        [[[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]]], dtype="float32"
    )
    old_layer = WhisperEncoderBlock(
        num_heads=2,
        intermediate_dim=8,
        dropout=0.0,
        name="transformer_encoder_layer_0",
    )
    _ = old_layer(hidden)
    new_model = build_encoder_block_model()
    copy_reference_weights(old_layer, new_model)
    new_output = ops.convert_to_numpy(new_model(hidden))
    old_output = ops.convert_to_numpy(old_layer(hidden))
    np.testing.assert_allclose(new_output, old_output, rtol=1e-5, atol=1e-5)


def test_decoder_block_matches_original():
    decoder_hidden = ops.convert_to_tensor(
        [[[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]]], dtype="float32"
    )
    encoder_hidden = ops.convert_to_tensor(
        [[[4.0, 3.0, 2.0, 1.0], [3.0, 2.0, 1.0, 0.0]]], dtype="float32"
    )
    decoder_padding_mask = ops.convert_to_tensor([[1, 1]], dtype="int32")
    old_layer = WhisperDecoderBlock(
        num_heads=2,
        intermediate_dim=8,
        dropout=0.0,
        name="transformer_decoder_layer_0",
    )
    _ = old_layer(decoder_hidden, encoder_hidden, decoder_padding_mask)
    new_model = build_decoder_block_model()
    copy_reference_weights(old_layer, new_model)
    new_output = ops.convert_to_numpy(
        new_model([decoder_hidden, encoder_hidden, decoder_padding_mask])
    )
    old_output = ops.convert_to_numpy(
        old_layer(
            decoder_hidden,
            encoder_hidden,
            decoder_padding_mask,
        )
    )
    np.testing.assert_allclose(new_output, old_output, rtol=1e-5, atol=1e-5)
