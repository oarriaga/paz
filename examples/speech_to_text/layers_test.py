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
from examples.speech_to_text.layers import mel_spectrogram
from examples.speech_to_text.layers import position_embedding
from examples.speech_to_text.layers import token_and_position_embedding
from examples.speech_to_text.weights import build_reference_whisper_audio_converter
from examples.speech_to_text.weights import build_reference_whisper_model
from examples.speech_to_text.weights import build_whisper_frontend_waveform
from examples.speech_to_text.weights import build_whisper_frontend_waveform_batch


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
        value = ops.convert_to_numpy(weight)
        weight_values[suffix] = value
        stripped_suffix = strip_transformer_prefix(suffix)
        weight_values[stripped_suffix] = value
        store_attention_aliases(weight_values, suffix, value)
    for weight in target_model.weights:
        suffix = build_weight_suffix(weight.path)
        weight.assign(weight_values[suffix])


def strip_transformer_prefix(path):
    path = path.replace("transformer_encoder_layer_0_", "")
    path = path.replace("transformer_decoder_layer_0_", "")
    return path


def store_attention_aliases(weight_values, suffix, value):
    if suffix.endswith("query/kernel"):
        weight_values["self_attention_layer_query/kernel"] = value
        weight_values["self_attention_query/kernel"] = value
        weight_values["cross_attention_query/kernel"] = value
    if suffix.endswith("query/bias"):
        weight_values["self_attention_layer_query/bias"] = value
        weight_values["self_attention_query/bias"] = value
        weight_values["cross_attention_query/bias"] = value
    if suffix.endswith("key/kernel"):
        weight_values["self_attention_layer_key/kernel"] = value
        weight_values["self_attention_key/kernel"] = value
        weight_values["cross_attention_key/kernel"] = value
    if suffix.endswith("value/kernel"):
        weight_values["self_attention_layer_value/kernel"] = value
        weight_values["self_attention_value/kernel"] = value
        weight_values["cross_attention_value/kernel"] = value
    if suffix.endswith("value/bias"):
        weight_values["self_attention_layer_value/bias"] = value
        weight_values["self_attention_value/bias"] = value
        weight_values["cross_attention_value/bias"] = value
    if suffix.endswith("attention_output/kernel"):
        weight_values["self_attention_layer_attention_output/kernel"] = value
        weight_values["self_attention_attention_output/kernel"] = value
        weight_values["cross_attention_attention_output/kernel"] = value
    if suffix.endswith("attention_output/bias"):
        weight_values["self_attention_layer_attention_output/bias"] = value
        weight_values["self_attention_attention_output/bias"] = value
        weight_values["cross_attention_attention_output/bias"] = value


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
    expected = np.array([[1.0, 2.0, 0.0, 0.0, 0.0]], dtype="float32")
    np.testing.assert_array_equal(ops.convert_to_numpy(new_waveform), expected)


def test_mel_filters_match_expected_shape():
    new_filters = build_mel_filters(80, 400, 16000, "float32")
    assert new_filters.shape == (201, 80)


def test_mel_spectrogram_matches_expected_shape():
    mel_filters = build_mel_filters(80, 400, 16000, "float32")
    mel_filters = ops.convert_to_tensor(mel_filters, dtype="float32")
    power_spectrogram = ops.ones((1, 2, 201), dtype="float32")
    outputs = mel_spectrogram(power_spectrogram, mel_filters)
    assert tuple(outputs.shape) == (1, 2, 80)


def test_frontend_matches_reference_example_values():
    waveform = ops.ones((2,), dtype="float32")
    mel_filters = build_mel_filters(80, 400, 100, "float32")
    mel_filters = ops.convert_to_tensor(mel_filters, dtype="float32")
    outputs = ops.convert_to_numpy(
        frontend(
            waveform,
            mel_filters,
            num_fft_bins=400,
            stride=100,
            sampling_rate=100,
            max_audio_length=5,
        )
    )
    expected = np.array(
        [1.1656, 1.0151, -0.8343, -0.8343, -0.8343], dtype="float32"
    )
    np.testing.assert_allclose(outputs[:, 0], expected, rtol=0.01, atol=0.01)


def test_frontend_matches_reference_single_waveform():
    reference_converter = build_reference_whisper_audio_converter(dtype="float32")
    waveform = build_whisper_frontend_waveform()
    mel_filters = build_mel_filters(80, 400, 16000, "float32")
    mel_filters = ops.convert_to_tensor(mel_filters, dtype="float32")
    new_output = ops.convert_to_numpy(frontend(waveform, mel_filters))
    reference_output = reference_converter(ops.convert_to_numpy(waveform)).numpy()
    np.testing.assert_allclose(new_output, reference_output, rtol=1e-5, atol=2e-5)


def test_frontend_matches_reference_batch():
    reference_converter = build_reference_whisper_audio_converter(dtype="float32")
    waveform_batch = build_whisper_frontend_waveform_batch()
    mel_filters = build_mel_filters(80, 400, 16000, "float32")
    mel_filters = ops.convert_to_tensor(mel_filters, dtype="float32")
    new_output = ops.convert_to_numpy(frontend(waveform_batch, mel_filters))
    reference_output = reference_converter(
        ops.convert_to_numpy(waveform_batch)
    ).numpy()
    np.testing.assert_allclose(new_output, reference_output, rtol=1e-5, atol=2e-5)


def test_position_embedding_matches_original():
    reference_model = build_reference_whisper_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    hidden = ops.ones((2, 3, 4), dtype="float32")
    reference_layer = reference_model.encoder_position_embedding
    new_model = build_position_model(trainable=False)
    copy_reference_weights(reference_layer, new_model)
    new_output = ops.convert_to_numpy(new_model(hidden))
    reference_output = ops.convert_to_numpy(reference_layer(hidden))
    np.testing.assert_allclose(
        new_output, reference_output, rtol=1e-5, atol=1e-5
    )


def test_position_embedding_with_positions_matches_original():
    reference_model = build_reference_whisper_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    hidden = ops.ones((2, 3, 4), dtype="float32")
    positions = ops.convert_to_tensor([[0, 1, 2], [2, 1, 0]], dtype="int32")
    reference_layer = reference_model.encoder_position_embedding
    new_model = build_position_with_positions_model(trainable=False)
    copy_reference_weights(reference_layer, new_model)
    new_output = ops.convert_to_numpy(new_model([hidden, positions]))
    reference_output = ops.convert_to_numpy(
        reference_layer(hidden, positions=positions)
    )
    np.testing.assert_allclose(
        new_output, reference_output, rtol=1e-5, atol=1e-5
    )


def test_token_and_position_embedding_matches_original():
    reference_model = build_reference_whisper_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    token_ids = ops.convert_to_tensor([[1, 2, 3]], dtype="int32")
    reference_layer = reference_model.decoder_embeddings
    new_model = build_token_and_position_model()
    copy_reference_weights(reference_layer, new_model)
    new_output = ops.convert_to_numpy(new_model(token_ids))
    reference_output = ops.convert_to_numpy(reference_layer(token_ids))
    np.testing.assert_allclose(
        new_output, reference_output, rtol=1e-5, atol=1e-5
    )


def test_decoder_mask_matches_original():
    decoder_sequence = ops.ones((1, 4, 4), dtype="float32")
    decoder_padding_mask = ops.convert_to_tensor([[1, 1, 1, 0]], dtype="int32")
    new_mask = ops.convert_to_numpy(
        build_decoder_self_attention_mask(decoder_sequence, decoder_padding_mask)
    )
    expected_mask = np.array(
        [[[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 0]]],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(new_mask, expected_mask)


def test_self_attention_matches_original():
    reference_model = build_reference_whisper_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    hidden = ops.convert_to_tensor(
        [[[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]]], dtype="float32"
    )
    reference_layer = reference_model.get_layer(
        "transformer_encoder_layer_0"
    )._self_attention_layer
    new_model = build_attention_model("self_attention_layer")
    copy_reference_weights(reference_layer, new_model)
    new_output = ops.convert_to_numpy(new_model([hidden, hidden]))
    reference_output = ops.convert_to_numpy(reference_layer(hidden, hidden))
    np.testing.assert_allclose(
        new_output, reference_output, rtol=1e-5, atol=1e-5
    )


def test_cross_attention_matches_original():
    reference_model = build_reference_whisper_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    decoder_hidden = ops.convert_to_tensor(
        [[[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]]], dtype="float32"
    )
    encoder_hidden = ops.convert_to_tensor(
        [[[4.0, 3.0, 2.0, 1.0], [3.0, 2.0, 1.0, 0.0]]], dtype="float32"
    )
    reference_layer = reference_model.get_layer(
        "transformer_decoder_layer_0"
    )._cross_attention_layer
    new_model = build_attention_model("cross_attention")
    copy_reference_weights(reference_layer, new_model)
    new_output = ops.convert_to_numpy(
        new_model([decoder_hidden, encoder_hidden])
    )
    reference_output = ops.convert_to_numpy(
        reference_layer(decoder_hidden, encoder_hidden)
    )
    np.testing.assert_allclose(
        new_output, reference_output, rtol=1e-5, atol=1e-5
    )


def test_encoder_block_matches_original():
    reference_model = build_reference_whisper_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    hidden = ops.convert_to_tensor(
        [[[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]]], dtype="float32"
    )
    reference_layer = reference_model.get_layer("transformer_encoder_layer_0")
    new_model = build_encoder_block_model()
    copy_reference_weights(reference_layer, new_model)
    new_output = ops.convert_to_numpy(new_model(hidden))
    reference_output = ops.convert_to_numpy(reference_layer(hidden))
    np.testing.assert_allclose(
        new_output, reference_output, rtol=1e-5, atol=1e-5
    )


def test_decoder_block_matches_original():
    reference_model = build_reference_whisper_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    decoder_hidden = ops.convert_to_tensor(
        [[[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]]], dtype="float32"
    )
    encoder_hidden = ops.convert_to_tensor(
        [[[4.0, 3.0, 2.0, 1.0], [3.0, 2.0, 1.0, 0.0]]], dtype="float32"
    )
    decoder_padding_mask = ops.convert_to_tensor([[1, 1]], dtype="int32")
    reference_layer = reference_model.get_layer("transformer_decoder_layer_0")
    new_model = build_decoder_block_model()
    copy_reference_weights(reference_layer, new_model)
    new_output = ops.convert_to_numpy(
        new_model([decoder_hidden, encoder_hidden, decoder_padding_mask])
    )
    reference_output = ops.convert_to_numpy(
        reference_layer(
            decoder_hidden,
            encoder_hidden,
            decoder_padding_mask=decoder_padding_mask,
        )
    )
    np.testing.assert_allclose(
        new_output, reference_output, rtol=1e-5, atol=1e-5
    )
