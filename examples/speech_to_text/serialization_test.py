from pathlib import Path

import numpy as np
from keras import ops
from keras.models import load_model

import examples.speech_to_text.weights as whisper_weights
from examples.speech_to_text.model2 import build_whisper_model_dir
from examples.speech_to_text.model2 import CONFIGS
from examples.speech_to_text.model2 import WHISPER_MODELS_DIR
from examples.speech_to_text.model2 import WhisperCrossCache
from examples.speech_to_text.model2 import WhisperDecoderStep
from examples.speech_to_text.model2 import WhisperEncoder


def test_exported_variant_names_match_available_local_presets():
    exported_variants = tuple(
        sorted(
            path.name
            for path in WHISPER_MODELS_DIR.iterdir()
            if (path / "encoder.weights.h5").exists()
        )
    )
    available_variants = tuple(
        sorted(
            variant_name
            for variant_name, _ in whisper_weights.find_available_whisper_presets()
        )
    )
    assert exported_variants == available_variants


def test_exported_base_en_artifact_directory_is_discoverable():
    assert build_whisper_model_dir("whisper_base_en").exists()


def test_exported_base_en_weight_files_exist():
    variant_dir = build_whisper_model_dir("whisper_base_en")
    paths = (
        variant_dir / "encoder.weights.h5",
        variant_dir / "cross_cache.weights.h5",
        variant_dir / "decoder_step.weights.h5",
    )
    assert all(path.exists() for path in paths)


def test_exported_tiny_runtime_models_load_from_keras_files(clear_keras_session):
    variant_dir = build_whisper_model_dir("whisper_tiny_en")
    paths = (
        variant_dir / "encoder.keras",
        variant_dir / "cross_cache.keras",
        variant_dir / "decoder_step.keras",
    )
    loaded_models = [load_model(Path(path), safe_mode=False) for path in paths]
    assert tuple(model.name for model in loaded_models) == (
        "whisper_tiny_en_encoder",
        "whisper_tiny_en_cross_cache",
        "whisper_tiny_en_decoder_step",
    )


def test_exported_base_en_encoder_matches_bridge(clear_keras_session):
    config = CONFIGS["whisper_base_en"]
    exported_encoder = WhisperEncoder(**config, weights="whisper_base_en")
    reference_encoder = WhisperEncoder(**config)
    whisper_weights.load_preset_weights(
        reference_encoder, "whisper_base_en", model_kind="encoder"
    )
    encoder_features, _, _ = whisper_weights.build_whisper_parity_inputs()
    exported_output = exported_encoder(encoder_features)
    reference_output = reference_encoder(encoder_features)
    assert np.allclose(
        ops.convert_to_numpy(exported_output),
        ops.convert_to_numpy(reference_output),
        rtol=1e-5,
        atol=1e-5,
    )


def test_exported_base_en_cross_cache_matches_bridge(clear_keras_session):
    config = CONFIGS["whisper_base_en"]
    exported_encoder = WhisperEncoder(**config, weights="whisper_base_en")
    reference_cross_cache = WhisperCrossCache(**config)
    whisper_weights.load_preset_weights(
        reference_cross_cache, "whisper_base_en", model_kind="cross_cache"
    )
    exported_cross_cache = WhisperCrossCache(**config, weights="whisper_base_en")
    encoder_features, _, _ = whisper_weights.build_whisper_parity_inputs()
    encoder_output = exported_encoder(encoder_features)
    exported_output = exported_cross_cache(encoder_output)
    reference_output = reference_cross_cache(encoder_output)
    assert np.allclose(
        ops.convert_to_numpy(exported_output),
        ops.convert_to_numpy(reference_output),
        rtol=1e-5,
        atol=1e-5,
    )


def test_exported_base_en_decoder_step_matches_bridge(clear_keras_session):
    config = CONFIGS["whisper_base_en"]
    exported_decoder_step = WhisperDecoderStep(
        **config, weights="whisper_base_en"
    )
    reference_decoder_step = WhisperDecoderStep(**config)
    whisper_weights.load_preset_weights(
        reference_decoder_step, "whisper_base_en", model_kind="decoder_step"
    )
    decoder_step_inputs = whisper_weights.build_decoder_step_inputs(
        "whisper_base_en"
    )
    exported_outputs = exported_decoder_step(decoder_step_inputs)
    reference_outputs = reference_decoder_step(decoder_step_inputs)
    logits_match = np.allclose(
        ops.convert_to_numpy(exported_outputs[0]),
        ops.convert_to_numpy(reference_outputs[0]),
        rtol=1e-5,
        atol=1e-5,
    )
    cache_match = np.allclose(
        ops.convert_to_numpy(exported_outputs[1]),
        ops.convert_to_numpy(reference_outputs[1]),
        rtol=1e-5,
        atol=1e-5,
    )
    assert logits_match and cache_match
