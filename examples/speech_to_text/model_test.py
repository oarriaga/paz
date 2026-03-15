import gc
from pathlib import Path

import keras
import numpy as np
import pytest
from keras import ops

import examples.speech_to_text.model as whisper_model
from examples.speech_to_text.model import find_whisper_variant_config
from examples.speech_to_text.model import Whisper
from examples.speech_to_text.model import WhisperBaseEn
from examples.speech_to_text.model import WhisperDecoder
from examples.speech_to_text.model import WhisperEncoder
from examples.speech_to_text.model import WhisperFrontend
from examples.speech_to_text.model import WHISPER_VARIANTS
from examples.speech_to_text.model import get_whisper_variant_names
from examples.speech_to_text.weights import build_missing_whisper_preset_message
from examples.speech_to_text.weights import build_reference_whisper_base_en_preset_model
from examples.speech_to_text.weights import build_reference_logits
from examples.speech_to_text.weights import build_whisper_base_en_parity_inputs
from examples.speech_to_text.weights import build_whisper_frontend_waveform
from examples.speech_to_text.weights import build_reference_whisper_model
from examples.speech_to_text.weights import call_reference_model
from examples.speech_to_text.weights import collect_logical_weight_path_names
from examples.speech_to_text.weights import collect_weight_path_names
from examples.speech_to_text.weights import copy_matching_weights
from examples.speech_to_text.weights import find_whisper_base_en_preset_dir
from examples.speech_to_text.weights import find_whisper_preset_dir
from examples.speech_to_text.weights import load_reference_whisper_preset_names


def test_encoder_model_matches_reference():
    reference_model = build_reference_whisper_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    clean_model = WhisperEncoder(1, 2, 4, 8, 80, 0.0, 6)
    encoder_features = ops.ones((1, 5, 80), dtype="float32")
    reference_model(
        {
            "encoder_features": encoder_features,
            "decoder_token_ids": ops.ones((1, 4), dtype="int32"),
            "decoder_padding_mask": ops.ones((1, 4), dtype="int32"),
        }
    )
    clean_model(encoder_features)
    clean_paths = collect_logical_weight_path_names(clean_model, "clean")
    reference_paths = collect_logical_weight_path_names(
        reference_model, "reference"
    )
    reference_paths = filter_encoder_paths(reference_paths)
    assert clean_paths == reference_paths


def test_model_param_count_matches_reference():
    reference_model = build_reference_whisper_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    clean_model = Whisper(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    encoder_features = ops.ones((1, 5, 80), dtype="float32")
    decoder_token_ids = ops.ones((1, 4), dtype="int32")
    decoder_padding_mask = ops.ones((1, 4), dtype="int32")
    call_reference_model(
        reference_model, encoder_features, decoder_token_ids, decoder_padding_mask
    )
    clean_model([encoder_features, decoder_token_ids, decoder_padding_mask])
    assert clean_model.count_params() == reference_model.count_params()


def test_decoder_model_matches_reference():
    reference_model = build_reference_whisper_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    clean_model = WhisperDecoder(10, 1, 2, 4, 8, 0.0, 6)
    encoder_features = ops.ones((1, 5, 80), dtype="float32")
    decoder_token_ids = ops.convert_to_tensor([[1, 2, 3, 4]], dtype="int32")
    decoder_padding_mask = ops.ones((1, 4), dtype="int32")
    reference_encoder, reference_decoder = call_reference_model(
        reference_model, encoder_features, decoder_token_ids, decoder_padding_mask
    )
    clean_model([decoder_token_ids, decoder_padding_mask, reference_encoder])
    clean_paths = collect_logical_weight_path_names(clean_model, "clean")
    reference_paths = collect_logical_weight_path_names(
        reference_model, "reference"
    )
    reference_paths = filter_decoder_paths(reference_paths)
    assert clean_paths == reference_paths


def test_small_reference_core_outputs_match_reference():
    reference_model = build_reference_whisper_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    clean_model = Whisper(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    encoder_features = ops.convert_to_tensor(
        np.arange(400.0).reshape((1, 5, 80)), dtype="float32"
    )
    decoder_token_ids = ops.convert_to_tensor([[1, 2, 3, 0]], dtype="int32")
    decoder_padding_mask = ops.convert_to_tensor([[1, 1, 1, 0]], dtype="int32")
    call_reference_model(
        reference_model, encoder_features, decoder_token_ids, decoder_padding_mask
    )
    clean_model([encoder_features, decoder_token_ids, decoder_padding_mask])
    copy_matching_weights(clean_model, reference_model)
    reference_outputs = call_reference_model(
        reference_model, encoder_features, decoder_token_ids, decoder_padding_mask
    )
    clean_outputs = clean_model(
        [encoder_features, decoder_token_ids, decoder_padding_mask]
    )
    clean_encoder = ops.convert_to_numpy(clean_outputs[0])
    clean_decoder = ops.convert_to_numpy(clean_outputs[1])
    reference_encoder = ops.convert_to_numpy(reference_outputs[0])
    reference_decoder = ops.convert_to_numpy(reference_outputs[1])
    equal = np.allclose(clean_encoder, reference_encoder, 1e-5, 1e-5)
    equal = equal and np.allclose(clean_decoder, reference_decoder, 1e-5, 1e-5)
    assert equal


def test_small_reference_logits_match_reference():
    reference_model = build_reference_whisper_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    clean_model = Whisper(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    encoder_features = ops.convert_to_tensor(
        np.arange(400.0).reshape((1, 5, 80)), dtype="float32"
    )
    decoder_token_ids = ops.convert_to_tensor([[1, 2, 3, 0]], dtype="int32")
    decoder_padding_mask = ops.convert_to_tensor([[1, 1, 1, 0]], dtype="int32")
    reference_outputs = call_reference_model(
        reference_model, encoder_features, decoder_token_ids, decoder_padding_mask
    )
    clean_model([encoder_features, decoder_token_ids, decoder_padding_mask])
    copy_matching_weights(clean_model, reference_model)
    reference_logits = build_reference_logits(reference_model, reference_outputs[1])
    clean_outputs = clean_model(
        [encoder_features, decoder_token_ids, decoder_padding_mask]
    )
    clean_logits = ops.convert_to_numpy(clean_outputs[2])
    reference_logits = ops.convert_to_numpy(reference_logits)
    np.testing.assert_allclose(clean_logits, reference_logits, rtol=1e-5, atol=1e-5)


def test_base_en_preset_encoder_matches_reference():
    clean_model, reference_model, model_inputs = build_base_en_preset_parity_state()
    reference_encoder, _ = call_reference_model(reference_model, *model_inputs)
    clean_encoder = clean_model(model_inputs)[0]
    np.testing.assert_allclose(
        ops.convert_to_numpy(clean_encoder),
        ops.convert_to_numpy(reference_encoder),
        rtol=1e-5,
        atol=1e-5,
    )


def test_base_en_preset_decoder_matches_reference():
    clean_model, reference_model, model_inputs = build_base_en_preset_parity_state()
    _, reference_decoder = call_reference_model(reference_model, *model_inputs)
    clean_decoder = clean_model(model_inputs)[1]
    np.testing.assert_allclose(
        ops.convert_to_numpy(clean_decoder),
        ops.convert_to_numpy(reference_decoder),
        rtol=1e-5,
        atol=1e-5,
    )


def test_base_en_preset_logits_match_reference():
    clean_model, reference_model, model_inputs = build_base_en_preset_parity_state()
    reference_decoder = call_reference_model(reference_model, *model_inputs)[1]
    reference_logits = build_reference_logits(reference_model, reference_decoder)
    clean_logits = clean_model(model_inputs)[2]
    np.testing.assert_allclose(
        ops.convert_to_numpy(clean_logits),
        ops.convert_to_numpy(reference_logits),
        rtol=1e-5,
        atol=1e-5,
    )


def test_runtime_files_do_not_import_keras_hub():
    root = Path(__file__).resolve().parent
    model_source = (root / "model.py").read_text(encoding="utf-8")
    layers_source = ""
    for path in sorted((root / "layers").glob("*.py")):
        layers_source += path.read_text(encoding="utf-8")
    result = "keras_hub" not in model_source and "keras_hub" not in layers_source
    assert result


def test_base_en_wrapper_matches_generic_builder():
    config = find_whisper_variant_config("whisper_base_en")
    wrapped_model = Whisper(**config)
    generic_model = Whisper(51864, 6, 8, 512, 2048, 80, 0.0, 3000, 448)
    encoder_features = ops.ones((1, 12, 80), dtype="float32")
    decoder_token_ids = ops.ones((1, 6), dtype="int32")
    decoder_padding_mask = ops.ones((1, 6), dtype="int32")
    wrapped_model([encoder_features, decoder_token_ids, decoder_padding_mask])
    generic_model([encoder_features, decoder_token_ids, decoder_padding_mask])
    result = collect_weight_path_names(wrapped_model) == collect_weight_path_names(
        generic_model
    )
    result = result and wrapped_model.count_params() == generic_model.count_params()
    assert result


def test_encoder_and_decoder_constructors_match_expected_shapes():
    wrapped_encoder_model = WhisperEncoder(6, 8, 512, 2048, 80, 0.0, 3000)
    wrapped_decoder_model = WhisperDecoder(51864, 6, 8, 512, 2048, 0.0, 448)
    encoder_features = ops.ones((1, 12, 80), dtype="float32")
    encoder_output = wrapped_encoder_model(encoder_features)
    decoder_token_ids = ops.ones((1, 6), dtype="int32")
    decoder_padding_mask = ops.ones((1, 6), dtype="int32")
    wrapped_decoder_model(
        [decoder_token_ids, decoder_padding_mask, encoder_output]
    )
    result = tuple(wrapped_encoder_model.output_shape) == (None, None, 512)
    result = result and tuple(wrapped_decoder_model.output_shape) == (
        None,
        None,
        512,
    )
    assert result


def test_frontend_constructor_has_expected_output_shape():
    wrapped_model = WhisperFrontend(80, 400, 100, 100, 5)
    waveform = ops.ones((1, 2), dtype="float32")
    wrapped_output = wrapped_model(waveform)
    assert tuple(wrapped_output.shape) == (1, 5, 80)


def test_base_en_waveform_frontend_output_fits_encoder_interface(
    clear_keras_session,
):
    frontend_model = WhisperFrontend()
    encoder_model = WhisperEncoder(6, 8, 512, 2048, 80, 0.0, 3000)
    waveform = ops.expand_dims(build_whisper_frontend_waveform(), axis=0)
    encoder_features = frontend_model(waveform)
    encoder_output = encoder_model(encoder_features)
    result = tuple(encoder_features.shape) == (1, 3000, 80)
    result = result and tuple(encoder_output.shape) == (1, 1500, 512)
    assert result


def test_logits_use_tied_decoder_embedding():
    model = Whisper(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    encoder_features = ops.convert_to_tensor(
        np.arange(400.0).reshape((1, 5, 80)), dtype="float32"
    )
    decoder_token_ids = ops.convert_to_tensor([[1, 2, 3, 0]], dtype="int32")
    decoder_padding_mask = ops.convert_to_tensor([[1, 1, 1, 0]], dtype="int32")
    encoder_output, decoder_output, logits = model(
        [encoder_features, decoder_token_ids, decoder_padding_mask]
    )
    embedding_layer = model.get_layer("decoder_token_embedding")
    manual_logits = embedding_layer(decoder_output, reverse=True)
    paths = collect_weight_path_names(model)
    result = np.allclose(
        ops.convert_to_numpy(logits),
        ops.convert_to_numpy(manual_logits),
        rtol=1e-5,
        atol=1e-5,
    )
    result = result and not any("logits" in path for path in paths)
    assert result


def test_all_whisper_variant_constructors_exist():
    preset_names = load_reference_whisper_preset_names()
    result = get_whisper_variant_names() == preset_names
    for variant_name in preset_names:
        result = result and hasattr(
            whisper_model, build_constructor_name(variant_name)
        )
    assert result


@pytest.mark.parametrize(
    (
        "variant_name,vocabulary_size,num_layers,num_heads,hidden_dim,"
        "intermediate_dim,num_mels,max_encoder_sequence_length,"
        "max_decoder_sequence_length,is_multilingual"
    ),
    WHISPER_VARIANTS,
)
def test_variant_random_models_build_and_match_metadata(
    clear_keras_session,
    variant_name,
    vocabulary_size,
    num_layers,
    num_heads,
    hidden_dim,
    intermediate_dim,
    num_mels,
    max_encoder_sequence_length,
    max_decoder_sequence_length,
    is_multilingual,
):
    config = find_whisper_variant_config(variant_name)
    core_builder = lambda: Whisper(**config)
    encoder_builder = lambda: WhisperEncoder(
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        num_mels,
        0.0,
        max_encoder_sequence_length,
    )
    decoder_builder = lambda: WhisperDecoder(
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        0.0,
        max_decoder_sequence_length,
    )
    core_input_shapes, core_output_shapes = build_core_model_shapes(core_builder)
    clear_model_session()
    encoder_input_shape, encoder_output_shape = build_encoder_model_shapes(
        encoder_builder
    )
    clear_model_session()
    decoder_input_shapes, decoder_output_shape = build_decoder_model_shapes(
        decoder_builder, hidden_dim
    )
    clear_model_session()
    result = len(core_input_shapes) == 3
    result = result and len(core_output_shapes) == 3
    result = result and core_input_shapes[0][-1] == num_mels
    result = result and core_output_shapes[0][-1] == hidden_dim
    result = result and core_output_shapes[1][-1] == hidden_dim
    result = result and core_output_shapes[2][-1] == vocabulary_size
    result = result and encoder_input_shape[-1] == num_mels
    result = result and encoder_output_shape[-1] == hidden_dim
    result = result and len(decoder_input_shapes) == 3
    result = result and decoder_input_shapes[2][-1] == hidden_dim
    result = result and decoder_output_shape[-1] == hidden_dim
    result = result and build_constructor_name(variant_name).startswith(
        "Whisper"
    )
    assert result


@pytest.mark.parametrize("variant_name", get_whisper_variant_names())
def test_variant_constructors_match_local_preset_availability(
    clear_keras_session, variant_name
):
    constructor_name = build_constructor_name(variant_name)
    constructor = getattr(whisper_model, constructor_name)
    preset_dir = find_whisper_preset_dir(variant_name)
    if preset_dir is None:
        with pytest.raises(FileNotFoundError):
            constructor()
        return
    assert constructor() is not None


def test_base_en_named_constructor_matches_config():
    config = find_whisper_variant_config("whisper_base_en")
    assert config == {
        "vocabulary_size": 51864,
        "num_layers": 6,
        "num_heads": 8,
        "hidden_dim": 512,
        "intermediate_dim": 2048,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    }


def filter_encoder_paths(paths):
    filtered = []
    for path in paths:
        if path.startswith("encoder_") or path.startswith("transformer_encoder_"):
            filtered.append(path)
    return filtered


def filter_decoder_paths(paths):
    filtered = []
    for path in paths:
        if path.startswith("decoder_") or path.startswith("transformer_decoder_"):
            filtered.append(path)
    return filtered


def build_constructor_name(variant_name):
    variant_name = variant_name.replace("whisper_", "")
    words = variant_name.split("_")
    words = [word.capitalize() for word in words]
    return "Whisper{}".format("".join(words))


def build_base_en_preset_parity_state():
    skip_if_base_en_preset_missing()
    reference_model = build_reference_whisper_base_en_preset_model()
    clean_model = WhisperBaseEn()
    model_inputs = build_whisper_base_en_parity_inputs()
    return clean_model, reference_model, model_inputs


def skip_if_base_en_preset_missing():
    if find_whisper_base_en_preset_dir() is None:
        pytest.skip(build_missing_whisper_preset_message("whisper_base_en"))


def build_core_model_shapes(builder):
    model = builder()
    input_shapes = tuple(tuple(shape) for shape in model.input_shape)
    output_shapes = tuple(tuple(shape) for shape in model.output_shape)
    del model
    return input_shapes, output_shapes


def build_encoder_model_shapes(builder):
    model = builder()
    input_shape = tuple(model.input_shape)
    output_shape = tuple(model.output_shape)
    del model
    return input_shape, output_shape


def build_decoder_model_shapes(builder, hidden_dim):
    model = builder()
    decoder_token_ids = ops.ones((1, 6), dtype="int32")
    decoder_padding_mask = ops.ones((1, 6), dtype="int32")
    encoder_output = ops.ones((1, 6, hidden_dim), dtype="float32")
    decoder_output = model(
        [decoder_token_ids, decoder_padding_mask, encoder_output]
    )
    input_shapes = tuple(tuple(shape) for shape in model.input_shape)
    output_shape = tuple(decoder_output.shape)
    del model
    return input_shapes, output_shape


def clear_model_session():
    keras.backend.clear_session()
    gc.collect()
