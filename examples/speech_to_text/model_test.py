import gc
import inspect
from pathlib import Path

import keras
import numpy as np
import pytest
from keras import ops

import examples.speech_to_text.model as whisper_model
from examples.speech_to_text.model import WHISPER_VARIANTS
from examples.speech_to_text.model import build_whisper_audio_frontend
from examples.speech_to_text.model import build_whisper_base_en_decoder_model
from examples.speech_to_text.model import build_whisper_base_en_encoder_model
from examples.speech_to_text.model import build_whisper_base_en_logits_model
from examples.speech_to_text.model import build_whisper_base_en_waveform_to_features_model
from examples.speech_to_text.model import build_whisper_core_logits_model
from examples.speech_to_text.model import build_whisper_core_model
from examples.speech_to_text.model import build_whisper_decoder_model
from examples.speech_to_text.model import build_whisper_encoder_model
from examples.speech_to_text.model import build_whisper_waveform_to_features_model
from examples.speech_to_text.model import get_whisper_variant_names
from examples.speech_to_text.weights import build_missing_whisper_preset_message
from examples.speech_to_text.weights import build_reference_whisper_base_en_preset_model
from examples.speech_to_text.weights import build_reference_logits
from examples.speech_to_text.weights import build_whisper_base_en_parity_inputs
from examples.speech_to_text.weights import build_whisper_frontend_waveform
from examples.speech_to_text.weights import build_reference_whisper_model
from examples.speech_to_text.weights import call_reference_model
from examples.speech_to_text.weights import collect_weight_path_names
from examples.speech_to_text.weights import copy_matching_weights
from examples.speech_to_text.weights import find_whisper_base_en_preset_dir
from examples.speech_to_text.weights import load_reference_whisper_preset_names


def test_encoder_model_matches_reference():
    reference_model = build_reference_whisper_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    clean_model = build_whisper_encoder_model(1, 2, 4, 8, 80, 0.0, 6)
    encoder_features = ops.ones((1, 5, 80), dtype="float32")
    reference_model(
        {
            "encoder_features": encoder_features,
            "decoder_token_ids": ops.ones((1, 4), dtype="int32"),
            "decoder_padding_mask": ops.ones((1, 4), dtype="int32"),
        }
    )
    clean_model(encoder_features)
    clean_paths = collect_weight_path_names(clean_model)
    reference_paths = collect_weight_path_names(reference_model)
    reference_paths = filter_encoder_paths(reference_paths)
    assert clean_paths == reference_paths


def test_model_param_count_matches_reference():
    reference_model = build_reference_whisper_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    clean_model = build_whisper_core_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
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
    clean_model = build_whisper_decoder_model(10, 1, 2, 4, 8, 0.0, 6)
    encoder_features = ops.ones((1, 5, 80), dtype="float32")
    decoder_token_ids = ops.convert_to_tensor([[1, 2, 3, 4]], dtype="int32")
    decoder_padding_mask = ops.ones((1, 4), dtype="int32")
    reference_encoder, reference_decoder = call_reference_model(
        reference_model, encoder_features, decoder_token_ids, decoder_padding_mask
    )
    clean_model([decoder_token_ids, decoder_padding_mask, reference_encoder])
    clean_paths = collect_weight_path_names(clean_model)
    reference_paths = collect_weight_path_names(reference_model)
    reference_paths = filter_decoder_paths(reference_paths)
    assert clean_paths == reference_paths


def test_small_reference_core_outputs_match_reference():
    reference_model = build_reference_whisper_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    clean_model = build_whisper_core_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
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
    clean_model = build_whisper_core_logits_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
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
    layers_source = (root / "layers.py").read_text(encoding="utf-8")
    result = "keras_hub" not in model_source and "keras_hub" not in layers_source
    assert result


def test_base_en_wrapper_matches_generic_builder():
    wrapped_model = build_whisper_base_en_logits_model()
    generic_model = build_whisper_core_logits_model(
        51864,
        6,
        8,
        512,
        2048,
        80,
        0.0,
        3000,
        448,
    )
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


def test_base_en_encoder_and_decoder_wrappers_match_generic_builders():
    wrapped_encoder_model = build_whisper_base_en_encoder_model()
    generic_encoder_model = build_whisper_encoder_model(6, 8, 512, 2048, 80)
    wrapped_decoder_model = build_whisper_base_en_decoder_model()
    generic_decoder_model = build_whisper_decoder_model(51864, 6, 8, 512, 2048)
    encoder_features = ops.ones((1, 12, 80), dtype="float32")
    encoder_output = wrapped_encoder_model(encoder_features)
    decoder_token_ids = ops.ones((1, 6), dtype="int32")
    decoder_padding_mask = ops.ones((1, 6), dtype="int32")
    wrapped_decoder_model(
        [decoder_token_ids, decoder_padding_mask, encoder_output]
    )
    generic_decoder_model(
        [decoder_token_ids, decoder_padding_mask, encoder_output]
    )
    result = collect_weight_path_names(
        wrapped_encoder_model
    ) == collect_weight_path_names(generic_encoder_model)
    result = result and wrapped_encoder_model.count_params() == (
        generic_encoder_model.count_params()
    )
    result = result and collect_weight_path_names(
        wrapped_decoder_model
    ) == collect_weight_path_names(generic_decoder_model)
    result = result and wrapped_decoder_model.count_params() == (
        generic_decoder_model.count_params()
    )
    assert result


def test_waveform_to_features_builder_matches_audio_frontend():
    wrapped_model = build_whisper_waveform_to_features_model(
        80, 400, 100, 100, 5
    )
    generic_model = build_whisper_audio_frontend(80, 400, 100, 100, 5)
    waveform = ops.ones((1, 2), dtype="float32")
    wrapped_output = wrapped_model(waveform)
    generic_output = generic_model(waveform)
    result = tuple(wrapped_output.shape) == tuple(generic_output.shape)
    result = result and wrapped_model.count_params() == generic_model.count_params()
    assert result


def test_base_en_waveform_frontend_matches_generic_builder():
    wrapped_model = build_whisper_base_en_waveform_to_features_model()
    generic_model = build_whisper_waveform_to_features_model(
        80, 400, 160, 16000, 30
    )
    waveform = ops.expand_dims(build_whisper_frontend_waveform(), axis=0)
    wrapped_output = wrapped_model(waveform)
    generic_output = generic_model(waveform)
    result = tuple(wrapped_output.shape) == tuple(generic_output.shape)
    result = result and wrapped_model.count_params() == generic_model.count_params()
    assert result


def test_base_en_waveform_frontend_output_fits_encoder_interface(
    clear_keras_session,
):
    frontend_model = build_whisper_base_en_waveform_to_features_model()
    encoder_model = build_whisper_encoder_model(6, 8, 512, 2048, 80, 0.0, 3000)
    waveform = ops.expand_dims(build_whisper_frontend_waveform(), axis=0)
    encoder_features = frontend_model(waveform)
    encoder_output = encoder_model(encoder_features)
    result = tuple(encoder_features.shape) == (1, 3000, 80)
    result = result and tuple(encoder_output.shape) == (1, 1500, 512)
    assert result


def test_logits_use_tied_decoder_embedding():
    model = build_whisper_core_logits_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    encoder_features = ops.convert_to_tensor(
        np.arange(400.0).reshape((1, 5, 80)), dtype="float32"
    )
    decoder_token_ids = ops.convert_to_tensor([[1, 2, 3, 0]], dtype="int32")
    decoder_padding_mask = ops.convert_to_tensor([[1, 1, 1, 0]], dtype="int32")
    encoder_output, decoder_output, logits = model(
        [encoder_features, decoder_token_ids, decoder_padding_mask]
    )
    embedding_layer = model.get_layer("decoder_token_and_position_embedding")
    manual_logits = embedding_layer.token_embedding(decoder_output, reverse=True)
    paths = collect_weight_path_names(model)
    result = np.allclose(
        ops.convert_to_numpy(logits),
        ops.convert_to_numpy(manual_logits),
        rtol=1e-5,
        atol=1e-5,
    )
    result = result and not any("logits" in path for path in paths)
    assert result


def test_all_whisper_variant_builders_exist():
    preset_names = load_reference_whisper_preset_names()
    result = get_whisper_variant_names() == preset_names
    for variant_name in preset_names:
        result = result and hasattr(whisper_model, "build_{}_model".format(variant_name))
        result = result and hasattr(
            whisper_model, "build_{}_logits_model".format(variant_name)
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
def test_variant_wrapper_uses_shared_variant_helper(
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
    builder = getattr(whisper_model, "build_{}_model".format(variant_name))
    logits_builder = getattr(
        whisper_model, "build_{}_logits_model".format(variant_name)
    )
    builder_source = inspect.getsource(builder)
    logits_source = inspect.getsource(logits_builder)
    result = "build_whisper_variant_model" in builder_source
    result = result and "build_whisper_variant_logits_model" in logits_source
    result = result and variant_name in builder_source
    result = result and variant_name in logits_source
    result = result and str(vocabulary_size) not in builder_source
    result = result and str(vocabulary_size) not in logits_source
    result = result and str(is_multilingual) not in builder_source
    result = result and str(is_multilingual) not in logits_source
    assert result


@pytest.mark.parametrize(
    (
        "variant_name,vocabulary_size,num_layers,num_heads,hidden_dim,"
        "intermediate_dim,num_mels,max_encoder_sequence_length,"
        "max_decoder_sequence_length,is_multilingual"
    ),
    WHISPER_VARIANTS,
)
def test_variant_models_build_and_match_metadata(
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
    core_builder = getattr(whisper_model, "build_{}_model".format(variant_name))
    logits_builder = getattr(
        whisper_model, "build_{}_logits_model".format(variant_name)
    )
    core_input_shapes, core_output_shapes = build_core_model_shapes(core_builder)
    clear_model_session()
    logits_input_shapes, logits_output_shapes, logits_paths, has_embedding = (
        build_logits_model_shapes(logits_builder)
    )
    result = len(core_input_shapes) == 3
    result = result and len(core_output_shapes) == 2
    result = result and core_input_shapes[0][-1] == num_mels
    result = result and core_output_shapes[0][-1] == hidden_dim
    result = result and core_output_shapes[1][-1] == hidden_dim
    result = result and len(logits_input_shapes) == 3
    result = result and len(logits_output_shapes) == 3
    result = result and logits_input_shapes[0][-1] == num_mels
    result = result and logits_output_shapes[0][-1] == hidden_dim
    result = result and logits_output_shapes[1][-1] == hidden_dim
    result = result and logits_output_shapes[2][-1] == vocabulary_size
    result = result and has_embedding
    result = result and not any("logits" in path for path in logits_paths)
    assert result


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


def build_base_en_preset_parity_state():
    skip_if_base_en_preset_missing()
    reference_model = build_reference_whisper_base_en_preset_model()
    clean_model = build_whisper_base_en_logits_model()
    model_inputs = build_whisper_base_en_parity_inputs()
    call_reference_model(reference_model, *model_inputs)
    clean_model(model_inputs)
    copy_matching_weights(clean_model, reference_model)
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


def build_logits_model_shapes(builder):
    model = builder()
    input_shapes = tuple(tuple(shape) for shape in model.input_shape)
    output_shapes = tuple(tuple(shape) for shape in model.output_shape)
    paths = collect_weight_path_names(model)
    has_embedding = model.get_layer("decoder_token_and_position_embedding") is not None
    del model
    return input_shapes, output_shapes, paths, has_embedding


def clear_model_session():
    keras.backend.clear_session()
    gc.collect()
