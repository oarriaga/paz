import pytest
from keras import ops

from examples.speech_to_text.model2 import CONFIGS
from examples.speech_to_text.model2 import WhisperCrossCache
from examples.speech_to_text.model2 import WhisperDecoderStep
from examples.speech_to_text.model2 import WhisperEncoder
from examples.speech_to_text.weights import find_variant_config
from examples.speech_to_text.weights import get_variant_names


def test_model2_config_names_match_reference_variant_names():
    assert tuple(CONFIGS.keys()) == get_variant_names()


@pytest.mark.parametrize("variant_name", get_variant_names())
def test_model2_config_matches_reference_config(variant_name):
    assert CONFIGS[variant_name] == find_variant_config(variant_name)


@pytest.mark.parametrize("variant_name", get_variant_names())
def test_model2_encoder_output_shape_matches_config(
    clear_keras_session, variant_name
):
    config = CONFIGS[variant_name]
    output_shape = WhisperEncoder(**config).output_shape
    assert output_shape == (None, None, config["hidden_dim"])


@pytest.mark.parametrize("variant_name", get_variant_names())
def test_model2_cross_cache_output_shape_matches_config(
    clear_keras_session, variant_name
):
    config = CONFIGS[variant_name]
    key_dim = config["hidden_dim"] // config["num_heads"]
    output_shape = WhisperCrossCache(**config).output_shape
    assert output_shape == (
        None,
        config["num_layers"],
        2,
        None,
        config["num_heads"],
        key_dim,
    )


@pytest.mark.parametrize("variant_name", get_variant_names())
def test_model2_decoder_step_logits_shape_matches_config(
    clear_keras_session, variant_name
):
    config = CONFIGS[variant_name]
    output_shapes = WhisperDecoderStep(**config).output_shape
    assert output_shapes[0] == (None, None, config["vocabulary_size"])


@pytest.mark.parametrize("variant_name", get_variant_names())
def test_model2_decoder_step_cache_shape_matches_config(
    clear_keras_session, variant_name
):
    config = CONFIGS[variant_name]
    key_dim = config["hidden_dim"] // config["num_heads"]
    output_shapes = WhisperDecoderStep(**config).output_shape
    assert output_shapes[1] == (
        None,
        config["num_layers"],
        2,
        None,
        config["num_heads"],
        key_dim,
    )


@pytest.mark.parametrize("variant_name", ["whisper_tiny_en", "whisper_tiny_multi"])
def test_model2_runtime_models_run_for_representative_variants(
    clear_keras_session, variant_name
):
    config = CONFIGS[variant_name]
    key_dim = config["hidden_dim"] // config["num_heads"]
    encoder = WhisperEncoder(**config)
    cross_cache_model = WhisperCrossCache(**config)
    decoder_step_model = WhisperDecoderStep(**config)
    encoder_features = ops.ones((1, 6, config["num_mels"]), dtype="float32")
    encoder_output = encoder(encoder_features)
    cross_attention_cache = cross_cache_model(encoder_output)
    self_attention_cache = ops.zeros(
        (1, config["num_layers"], 2, 4, config["num_heads"], key_dim),
        dtype="float32",
    )
    decoder_token_ids = ops.ones((1, 1), dtype="int32")
    cache_update_index = ops.convert_to_tensor([0], dtype="int32")
    output_shapes = tuple(
        tuple(value.shape)
        for value in decoder_step_model(
            [
                decoder_token_ids,
                self_attention_cache,
                cross_attention_cache,
                cache_update_index,
            ]
        )
    )
    assert output_shapes == (
        (1, 1, config["vocabulary_size"]),
        (1, config["num_layers"], 2, 4, config["num_heads"], key_dim),
    )
