import inspect

import numpy as np
import pytest
from keras import ops

import examples.speech_to_text.weights as whisper_weights
from examples.speech_to_text.model import Whisper
from examples.speech_to_text.weights import find_variant_config
from examples.speech_to_text.weights import get_variant_names
from examples.speech_to_text.weights import VARIANTS
from examples.speech_to_text.weights import build_missing_whisper_preset_message
from examples.speech_to_text.weights import build_reference_whisper_base_en_model
from examples.speech_to_text.weights import build_reference_whisper_base_en_preset_model
from examples.speech_to_text.weights import build_whisper_base_en_parity_inputs
from examples.speech_to_text.weights import build_whisper_parity_inputs
from examples.speech_to_text.weights import call_reference_model
from examples.speech_to_text.weights import collect_weight_path_names
from examples.speech_to_text.weights import copy_matching_weights
from examples.speech_to_text.weights import count_params_from_weights
from examples.speech_to_text.weights import find_duplicate_weight_paths
from examples.speech_to_text.weights import find_available_whisper_presets
from examples.speech_to_text.weights import find_whisper_base_en_preset_dir
from examples.speech_to_text.weights import find_whisper_preset_dir
from examples.speech_to_text.weights import build_reference_logits
from examples.speech_to_text.weights import build_reference_whisper_preset_model
from examples.speech_to_text.weights import collect_logical_weight_path_names
from examples.speech_to_text.weights import load_reference_whisper_preset_names
from examples.speech_to_text.weights import preset_matches_base_en_arguments


def test_base_en_argument_count_is_expected():
    model = build_reference_whisper_base_en_model()
    assert model.count_params() == 72593408


def test_reference_weight_inventory_is_stable():
    model = build_reference_whisper_base_en_model()
    paths = collect_weight_path_names(model)
    expected = [
        "decoder_layer_norm/beta",
        "decoder_layer_norm/gamma",
        "decoder_token_and_position_embedding/position_embedding/embeddings",
        "decoder_token_and_position_embedding/token_embedding/embeddings",
        "encoder_layer_norm/beta",
        "encoder_layer_norm/gamma",
        "encoder_position_embedding/embeddings",
        "encoder_token_embedding_conv_layer_1/bias",
        "encoder_token_embedding_conv_layer_1/kernel",
        "encoder_token_embedding_conv_layer_2/bias",
        "encoder_token_embedding_conv_layer_2/kernel",
    ]
    assert paths[:11] == expected


def test_reference_key_projection_bias_is_absent():
    model = build_reference_whisper_base_en_model()
    encoder_layer = model.get_layer("transformer_encoder_layer_0")
    decoder_layer = model.get_layer("transformer_decoder_layer_0")
    result = (
        encoder_layer._self_attention_layer._key_dense.bias is None
        and decoder_layer._self_attention_layer._key_dense.bias is None
        and decoder_layer._cross_attention_layer._key_dense.bias is None
    )
    assert result


def test_reference_param_count_matches_weight_count():
    model = build_reference_whisper_base_en_model()
    assert count_params_from_weights(model) == model.count_params()


def test_all_reference_whisper_variant_builders_exist():
    preset_names = load_reference_whisper_preset_names()
    result = get_variant_names() == preset_names
    for variant_name in preset_names:
        builder_name = "build_reference_{}_model".format(variant_name)
        result = result and hasattr(whisper_weights, builder_name)
    assert result


@pytest.mark.parametrize(
    (
        "variant_name,vocabulary_size,num_layers,num_heads,hidden_dim,"
        "intermediate_dim,num_mels,max_encoder_sequence_length,"
        "max_decoder_sequence_length,is_multilingual"
    ),
    VARIANTS,
)
def test_reference_variant_wrapper_uses_shared_helper(
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
    builder = getattr(
        whisper_weights, "build_reference_{}_model".format(variant_name)
    )
    source = inspect.getsource(builder)
    result = "build_reference_whisper_variant_model" in source
    result = result and variant_name in source
    result = result and str(vocabulary_size) not in source
    result = result and str(is_multilingual) not in source
    assert result


@pytest.mark.parametrize(
    (
        "variant_name,vocabulary_size,num_layers,num_heads,hidden_dim,"
        "intermediate_dim,num_mels,max_encoder_sequence_length,"
        "max_decoder_sequence_length,is_multilingual"
    ),
    VARIANTS,
)
def test_variant_preset_config_matches_expected_arguments(
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
    preset_dir = find_whisper_preset_dir(variant_name)
    if preset_dir is None:
        pytest.skip(build_missing_whisper_preset_message(variant_name))
    assert whisper_weights.preset_matches_variant_arguments(variant_name, preset_dir)


def test_available_local_presets_are_known_variants():
    available_variants = tuple(
        variant_name for variant_name, _ in find_available_whisper_presets()
    )
    result = all(
        variant_name in get_variant_names()
        for variant_name in available_variants
    )
    assert result


def test_base_en_preset_config_matches_expected_arguments():
    preset_dir = require_base_en_preset_dir()
    assert preset_matches_base_en_arguments(preset_dir)


def test_base_en_preset_is_discoverable():
    assert require_base_en_preset_dir() is not None


def test_base_en_preset_weight_inventory_matches_clean_model():
    clean_model, reference_model = build_base_en_preset_weight_pair()
    assert len(collect_weight_path_names(clean_model)) == len(
        collect_weight_path_names(reference_model)
    )


def test_base_en_preset_weight_paths_align_logically():
    clean_model, reference_model = build_base_en_preset_weight_pair()
    assert collect_logical_weight_path_names(
        clean_model, "clean"
    ) == collect_logical_weight_path_names(
        reference_model, "reference"
    )


@pytest.mark.parametrize(
    (
        "variant_name,vocabulary_size,num_layers,num_heads,hidden_dim,"
        "intermediate_dim,num_mels,max_encoder_sequence_length,"
        "max_decoder_sequence_length,is_multilingual"
    ),
    VARIANTS,
)
def test_available_non_base_presets_match_reference(
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
    if variant_name == "whisper_base_en":
        pytest.skip("whisper_base_en has dedicated preset parity tests.")
    preset_dir = find_whisper_preset_dir(variant_name)
    if preset_dir is None:
        pytest.skip(build_missing_whisper_preset_message(variant_name))
    reference_model = build_reference_whisper_preset_model(variant_name)
    clean_model = Whisper(**find_variant_config(variant_name))
    encoder_features, decoder_token_ids, decoder_padding_mask = (
        build_whisper_parity_inputs()
    )
    reference_outputs = call_reference_model(
        reference_model, encoder_features, decoder_token_ids, decoder_padding_mask
    )
    clean_model([encoder_features, decoder_token_ids, decoder_padding_mask])
    copy_matching_weights(clean_model, reference_model)
    reference_logits = build_reference_logits(reference_model, reference_outputs[1])
    clean_outputs = clean_model(
        [encoder_features, decoder_token_ids, decoder_padding_mask]
    )
    result = np.allclose(
        ops.convert_to_numpy(clean_outputs[0]),
        ops.convert_to_numpy(reference_outputs[0]),
        rtol=1e-5,
        atol=1e-5,
    )
    result = result and np.allclose(
        ops.convert_to_numpy(clean_outputs[1]),
        ops.convert_to_numpy(reference_outputs[1]),
        rtol=1e-5,
        atol=1e-5,
    )
    result = result and np.allclose(
        ops.convert_to_numpy(clean_outputs[2]),
        ops.convert_to_numpy(reference_logits),
        rtol=1e-5,
        atol=1e-5,
    )
    assert result


def test_copy_matching_weights_reports_count_mismatch():
    clean_model = DummyModel([DummyWeight("a", (1,))])
    reference_model = DummyModel(
        [DummyWeight("a", (1,)), DummyWeight("b", (1,))]
    )
    with pytest.raises(
        ValueError,
        match="Clean-only paths: \\[\\]. Reference-only paths: \\['b'\\]",
    ):
        copy_matching_weights(clean_model, reference_model)


def test_copy_matching_weights_reports_path_mismatch():
    clean_model = DummyModel(
        [DummyWeight("a", (1,)), DummyWeight("c", (1,))]
    )
    reference_model = DummyModel(
        [DummyWeight("a", (1,)), DummyWeight("b", (1,))]
    )
    with pytest.raises(ValueError, match="Weight path mismatch at index 1"):
        copy_matching_weights(clean_model, reference_model)


def test_copy_matching_weights_reports_shape_mismatch():
    clean_model = DummyModel([DummyWeight("a", (1, 2))])
    reference_model = DummyModel([DummyWeight("a", (2, 1))])
    with pytest.raises(
        ValueError, match="clean=\\(1, 2\\) reference=\\(2, 1\\)"
    ):
        copy_matching_weights(clean_model, reference_model)


def test_find_duplicate_weight_paths_reports_duplicates():
    duplicates = find_duplicate_weight_paths(
        [("a", object()), ("a", object()), ("b", object())]
    )
    assert duplicates == ["a"]


def test_copy_matching_weights_reports_duplicate_clean_paths():
    clean_model = DummyModel(
        [DummyWeight("a", (1,)), DummyWeight("a", (1,))]
    )
    reference_model = DummyModel(
        [DummyWeight("a", (1,)), DummyWeight("b", (1,))]
    )
    with pytest.raises(ValueError, match="Duplicate clean weight paths found"):
        copy_matching_weights(clean_model, reference_model)


def require_base_en_preset_dir():
    preset_dir = find_whisper_base_en_preset_dir()
    if preset_dir is None:
        pytest.skip(build_missing_whisper_preset_message("whisper_base_en"))
    return preset_dir


def build_base_en_preset_weight_pair():
    require_base_en_preset_dir()
    reference_model = build_reference_whisper_base_en_preset_model()
    clean_model = Whisper(**find_variant_config("whisper_base_en"))
    model_inputs = build_whisper_base_en_parity_inputs()
    call_reference_model(reference_model, *model_inputs)
    clean_model(model_inputs)
    return clean_model, reference_model


class DummyWeight:
    def __init__(self, path, shape):
        self.path = path
        self.shape = shape
        self.value = None

    def assign(self, value):
        self.value = value


class DummyModel:
    def __init__(self, weights):
        self.weights = weights
