import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import keras
import numpy as np

from examples.speech_to_text.model import build_whisper_base_en_decoder_model
from examples.speech_to_text.model import build_whisper_base_en_encoder_model
from examples.speech_to_text.model import build_whisper_base_en_logits_model
from examples.speech_to_text.model import find_whisper_variant_values
from examples.speech_to_text.model import get_whisper_variant_names


WHISPER_BASE_EN_PRESET_ENV = "PAZ_WHISPER_BASE_EN_PRESET_DIR"


def build_reference_whisper_model(
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
    name="reference_whisper",
):
    WhisperBackbone = load_reference_whisper_backbone()
    return WhisperBackbone(
        vocabulary_size=vocabulary_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_mels=num_mels,
        dropout=dropout,
        max_encoder_sequence_length=max_encoder_sequence_length,
        max_decoder_sequence_length=max_decoder_sequence_length,
        dtype=dtype,
        name=name,
    )


def build_reference_whisper_variant_model(
    variant_name, dtype="float32", name=None
):
    variant_values = find_whisper_variant_values(variant_name)
    if name is None:
        name = "reference_{}".format(variant_name)
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
    return build_reference_whisper_model(
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


def build_reference_whisper_audio_converter(
    num_mels=80,
    num_fft_bins=400,
    stride=160,
    sampling_rate=16000,
    max_audio_length=30,
    dtype="float32",
    name="reference_whisper_audio_converter",
):
    WhisperAudioConverter = load_reference_whisper_audio_converter()
    return WhisperAudioConverter(
        num_mels=num_mels,
        num_fft_bins=num_fft_bins,
        stride=stride,
        sampling_rate=sampling_rate,
        max_audio_length=max_audio_length,
        dtype=dtype,
        name=name,
    )


def build_reference_whisper_base_en_model(
    dtype="float32", name="reference_whisper_base_en"
):
    return build_reference_whisper_variant_model("whisper_base_en", dtype, name)


def build_reference_whisper_tiny_en_model(
    dtype="float32", name="reference_whisper_tiny_en"
):
    return build_reference_whisper_variant_model("whisper_tiny_en", dtype, name)


def build_reference_whisper_small_en_model(
    dtype="float32", name="reference_whisper_small_en"
):
    return build_reference_whisper_variant_model("whisper_small_en", dtype, name)


def build_reference_whisper_medium_en_model(
    dtype="float32", name="reference_whisper_medium_en"
):
    return build_reference_whisper_variant_model("whisper_medium_en", dtype, name)


def build_reference_whisper_tiny_multi_model(
    dtype="float32", name="reference_whisper_tiny_multi"
):
    return build_reference_whisper_variant_model("whisper_tiny_multi", dtype, name)


def build_reference_whisper_base_multi_model(
    dtype="float32", name="reference_whisper_base_multi"
):
    return build_reference_whisper_variant_model("whisper_base_multi", dtype, name)


def build_reference_whisper_small_multi_model(
    dtype="float32", name="reference_whisper_small_multi"
):
    return build_reference_whisper_variant_model("whisper_small_multi", dtype, name)


def build_reference_whisper_medium_multi_model(
    dtype="float32", name="reference_whisper_medium_multi"
):
    return build_reference_whisper_variant_model("whisper_medium_multi", dtype, name)


def build_reference_whisper_large_multi_model(
    dtype="float32", name="reference_whisper_large_multi"
):
    return build_reference_whisper_variant_model("whisper_large_multi", dtype, name)


def build_reference_whisper_large_multi_v2_model(
    dtype="float32", name="reference_whisper_large_multi_v2"
):
    return build_reference_whisper_variant_model(
        "whisper_large_multi_v2", dtype, name
    )


def build_reference_whisper_base_en_preset_model(
    dtype="float32", name="reference_whisper_base_en_preset"
):
    preset_dir = require_whisper_preset_dir("whisper_base_en")
    return build_reference_whisper_model_from_preset_dir(
        preset_dir, dtype=dtype, name=name
    )


def build_preset_loaded_whisper_base_en_logits_model(
    dtype="float32", name="whisper_base_en_logits"
):
    reference_model = build_reference_whisper_base_en_preset_model(dtype=dtype)
    clean_model = build_whisper_base_en_logits_model(dtype=dtype, name=name)
    model_inputs = build_whisper_base_en_parity_inputs()
    call_reference_model(reference_model, *model_inputs)
    clean_model(model_inputs)
    copy_matching_weights(clean_model, reference_model)
    return clean_model


def build_preset_loaded_whisper_base_en_encoder_model(
    dtype="float32", name="whisper_base_en_encoder"
):
    reference_model = build_reference_whisper_base_en_preset_model(dtype=dtype)
    clean_model = build_whisper_base_en_encoder_model(dtype=dtype, name=name)
    encoder_features, decoder_token_ids, decoder_padding_mask = (
        build_whisper_base_en_parity_inputs()
    )
    call_reference_model(
        reference_model,
        encoder_features,
        decoder_token_ids,
        decoder_padding_mask,
    )
    clean_model(encoder_features)
    copy_matching_encoder_weights(clean_model, reference_model)
    return clean_model


def build_preset_loaded_whisper_base_en_decoder_model(
    dtype="float32", name="whisper_base_en_decoder"
):
    reference_model = build_reference_whisper_base_en_preset_model(dtype=dtype)
    clean_model = build_whisper_base_en_decoder_model(dtype=dtype, name=name)
    encoder_features, decoder_token_ids, decoder_padding_mask = (
        build_whisper_base_en_parity_inputs()
    )
    reference_encoder_output = call_reference_model(
        reference_model,
        encoder_features,
        decoder_token_ids,
        decoder_padding_mask,
    )[0]
    clean_model(
        [decoder_token_ids, decoder_padding_mask, reference_encoder_output]
    )
    copy_matching_decoder_weights(clean_model, reference_model)
    return clean_model


def build_reference_whisper_preset_model(
    variant_name, dtype="float32", name=None
):
    preset_dir = require_whisper_preset_dir(variant_name)
    if name is None:
        name = "reference_{}_preset".format(variant_name)
    return build_reference_whisper_model_from_preset_dir(
        preset_dir, dtype=dtype, name=name
    )


def build_whisper_parity_inputs():
    encoder_features = np.arange(960, dtype="float32").reshape((1, 12, 80))
    encoder_features = (encoder_features - 480.0) / 100.0
    decoder_token_ids = np.array(
        [[50257, 50362, 464, 2068, 7586, 0]], dtype="int32"
    )
    decoder_padding_mask = np.array([[1, 1, 1, 1, 1, 0]], dtype="int32")
    return (
        keras.ops.convert_to_tensor(encoder_features),
        keras.ops.convert_to_tensor(decoder_token_ids, dtype="int32"),
        keras.ops.convert_to_tensor(decoder_padding_mask, dtype="int32"),
    )


def build_whisper_frontend_waveform(num_samples=3200):
    sample_positions = np.arange(num_samples, dtype="float32")
    waveform = np.sin(2.0 * np.pi * sample_positions / 160.0)
    waveform += 0.5 * np.cos(2.0 * np.pi * sample_positions / 90.0)
    envelope = np.linspace(-1.0, 1.0, num_samples, dtype="float32")
    waveform = waveform * envelope
    return keras.ops.convert_to_tensor(waveform, dtype="float32")


def build_whisper_frontend_waveform_batch(num_samples=3200):
    first_waveform = keras.ops.convert_to_numpy(
        build_whisper_frontend_waveform(num_samples)
    )
    second_waveform = np.roll(first_waveform[::-1], 37)
    waveform_batch = np.stack([first_waveform, second_waveform], axis=0)
    return keras.ops.convert_to_tensor(waveform_batch, dtype="float32")


def build_whisper_base_en_parity_inputs():
    return build_whisper_parity_inputs()


def call_reference_model(
    model, encoder_features, decoder_token_ids, decoder_padding_mask
):
    outputs = model(
        {
            "encoder_features": encoder_features,
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }
    )
    return (
        outputs["encoder_sequence_output"],
        outputs["decoder_sequence_output"],
    )


def build_reference_logits(reference_model, decoder_hidden_states):
    token_embedding = reference_model.decoder_embeddings.token_embedding
    return token_embedding(decoder_hidden_states, reverse=True)


def collect_weight_paths(model):
    weights = []
    for weight in model.weights:
        weights.append((weight.path, weight))
    weights.sort(key=lambda pair: pair[0])
    return weights


def collect_weight_path_names(model):
    return [path for path, _ in collect_weight_paths(model)]


def collect_logical_weight_path_names(model, label):
    return [path for path, _ in collect_logical_weight_paths(model, label)]


def copy_matching_weights(clean_model, reference_model):
    clean_weights = collect_logical_weight_paths(clean_model, "clean")
    reference_weights = collect_logical_weight_paths(reference_model, "reference")
    validate_unique_weight_paths(clean_weights, "clean")
    validate_unique_weight_paths(reference_weights, "reference")
    validate_weight_count_alignment(clean_weights, reference_weights)
    validate_weight_path_alignment(clean_weights, reference_weights)
    copied_paths = []
    for clean_pair, reference_pair in zip(clean_weights, reference_weights):
        clean_path, clean_weight = clean_pair
        _, reference_weight = reference_pair
        validate_weight_shape_alignment(clean_pair, reference_pair)
        clean_weight.assign(reference_weight)
        copied_paths.append(clean_path)
    return copied_paths


def copy_matching_encoder_weights(clean_model, reference_model):
    clean_weights = filter_encoder_weight_pairs(
        collect_logical_weight_paths(clean_model, "clean")
    )
    reference_weights = filter_encoder_weight_pairs(
        collect_logical_weight_paths(reference_model, "reference")
    )
    validate_unique_weight_paths(clean_weights, "clean")
    validate_unique_weight_paths(reference_weights, "reference")
    validate_weight_count_alignment(clean_weights, reference_weights)
    validate_weight_path_alignment(clean_weights, reference_weights)
    copied_paths = []
    for clean_pair, reference_pair in zip(clean_weights, reference_weights):
        clean_path, clean_weight = clean_pair
        _, reference_weight = reference_pair
        validate_weight_shape_alignment(clean_pair, reference_pair)
        clean_weight.assign(reference_weight)
        copied_paths.append(clean_path)
    return copied_paths


def copy_matching_decoder_weights(clean_model, reference_model):
    clean_weights = filter_decoder_weight_pairs(
        collect_logical_weight_paths(clean_model, "clean")
    )
    reference_weights = filter_decoder_weight_pairs(
        collect_logical_weight_paths(reference_model, "reference")
    )
    validate_unique_weight_paths(clean_weights, "clean")
    validate_unique_weight_paths(reference_weights, "reference")
    validate_weight_count_alignment(clean_weights, reference_weights)
    validate_weight_path_alignment(clean_weights, reference_weights)
    copied_paths = []
    for clean_pair, reference_pair in zip(clean_weights, reference_weights):
        clean_path, clean_weight = clean_pair
        _, reference_weight = reference_pair
        validate_weight_shape_alignment(clean_pair, reference_pair)
        clean_weight.assign(reference_weight)
        copied_paths.append(clean_path)
    return copied_paths


def count_params_from_weights(model):
    total = 0
    for _, weight in collect_weight_paths(model):
        total += int(np.prod(weight.shape))
    return total


def collect_logical_weight_paths(model, label):
    logical_weights = []
    for path, weight in collect_weight_paths(model):
        logical_path = normalize_weight_path(path, label)
        logical_weights.append((logical_path, weight))
    logical_weights.sort(key=lambda pair: pair[0])
    return logical_weights


def filter_encoder_weight_pairs(weight_pairs):
    filtered_pairs = []
    for path, weight in weight_pairs:
        if path.startswith("encoder_") or path.startswith("transformer_encoder_"):
            filtered_pairs.append((path, weight))
    return filtered_pairs


def filter_decoder_weight_pairs(weight_pairs):
    filtered_pairs = []
    for path, weight in weight_pairs:
        if path.startswith("decoder_") or path.startswith("transformer_decoder_"):
            filtered_pairs.append((path, weight))
    return filtered_pairs


def normalize_weight_path(path, label):
    if label == "reference":
        return path
    if label == "clean":
        return normalize_clean_weight_path(path)
    raise ValueError("Unknown weight path label: {}".format(label))


def normalize_clean_weight_path(path):
    path = path.replace(
        "decoder_token_embedding/",
        "decoder_token_and_position_embedding/token_embedding/",
    )
    path = path.replace(
        "decoder_position_embedding/",
        "decoder_token_and_position_embedding/position_embedding/",
    )
    path = path.replace(
        "_self_attention_layer_query/",
        "/self_attention_layer/query/",
    )
    path = path.replace(
        "_self_attention_layer_key/",
        "/self_attention_layer/key/",
    )
    path = path.replace(
        "_self_attention_layer_value/",
        "/self_attention_layer/value/",
    )
    path = path.replace(
        "_self_attention_layer_attention_output/",
        "/self_attention_layer/attention_output/",
    )
    path = path.replace(
        "_self_attention_layer_norm/",
        "/self_attention_layer_norm/",
    )
    path = path.replace(
        "_self_attention_query/",
        "/self_attention/query/",
    )
    path = path.replace(
        "_self_attention_key/",
        "/self_attention/key/",
    )
    path = path.replace(
        "_self_attention_value/",
        "/self_attention/value/",
    )
    path = path.replace(
        "_self_attention_attention_output/",
        "/self_attention/attention_output/",
    )
    path = path.replace(
        "_cross_attention_query/",
        "/cross_attention/query/",
    )
    path = path.replace(
        "_cross_attention_key/",
        "/cross_attention/key/",
    )
    path = path.replace(
        "_cross_attention_value/",
        "/cross_attention/value/",
    )
    path = path.replace(
        "_cross_attention_attention_output/",
        "/cross_attention/attention_output/",
    )
    path = path.replace(
        "_cross_attention_layer_norm/",
        "/cross_attention_layer_norm/",
    )
    path = path.replace(
        "_feedforward_layer_norm/",
        "/feedforward_layer_norm/",
    )
    path = path.replace(
        "_feedforward_intermediate_dense/",
        "/feedforward_intermediate_dense/",
    )
    path = path.replace(
        "_feedforward_output_dense/",
        "/feedforward_output_dense/",
    )
    return path


def build_reference_whisper_model_from_preset_dir(
    preset_dir, dtype="float32", name=None
):
    preset_dir = Path(preset_dir)
    ensure_reference_modules()
    config = load_preset_config(preset_dir)
    config = dict(config)
    config["config"] = dict(config["config"])
    if dtype is not None:
        config["config"]["dtype"] = dtype
    if name is not None:
        config["config"]["name"] = name
    model = keras.saving.deserialize_keras_object(config)
    model.load_weights(str(find_preset_weights_path(preset_dir)))
    return model


def load_preset_config(preset_dir):
    config_path = Path(preset_dir) / "config.json"
    with open(config_path, encoding="utf-8") as config_file:
        return json.load(config_file)


def preset_matches_base_en_arguments(preset_dir):
    return preset_matches_variant_arguments("whisper_base_en", preset_dir)


def preset_matches_variant_arguments(variant_name, preset_dir):
    variant_values = find_whisper_variant_values(variant_name)
    return preset_matches_expected_arguments(
        preset_dir,
        variant_values[1],
        variant_values[2],
        variant_values[3],
        variant_values[4],
        variant_values[5],
        variant_values[6],
        variant_values[7],
        variant_values[8],
    )


def preset_matches_expected_arguments(
    preset_dir,
    vocabulary_size,
    num_layers,
    num_heads,
    hidden_dim,
    intermediate_dim,
    num_mels=80,
    max_encoder_sequence_length=3000,
    max_decoder_sequence_length=448,
):
    config = load_preset_config(preset_dir)
    config = config["config"]
    result = config["vocabulary_size"] == vocabulary_size
    result = result and config["num_layers"] == num_layers
    result = result and config["num_heads"] == num_heads
    result = result and config["hidden_dim"] == hidden_dim
    result = result and config["intermediate_dim"] == intermediate_dim
    result = result and config["num_mels"] == num_mels
    result = result and (
        config["max_encoder_sequence_length"] == max_encoder_sequence_length
    )
    result = result and (
        config["max_decoder_sequence_length"] == max_decoder_sequence_length
    )
    return result


def find_whisper_base_en_preset_dir():
    return find_whisper_preset_dir("whisper_base_en")


def find_whisper_preset_dir(variant_name):
    candidates = build_preset_dir_candidates(variant_name)
    for candidate in candidates:
        candidate = Path(candidate).expanduser()
        if has_whisper_preset_files(candidate):
            return candidate
    return None


def find_available_whisper_presets():
    available_presets = []
    for variant_name in get_whisper_variant_names():
        preset_dir = find_whisper_preset_dir(variant_name)
        if preset_dir is not None:
            available_presets.append((variant_name, preset_dir))
    return tuple(available_presets)


def build_preset_dir_candidates(variant_name):
    candidates = []
    candidates.extend(build_kagglehub_preset_dir_candidates(variant_name))
    cache_root = Path.home() / ".keras" / "models"
    if cache_root.exists():
        pattern = "*{}*".format(variant_name)
        for candidate in sorted(cache_root.glob(pattern)):
            candidates.append(candidate)
    env_path = os.environ.get(build_whisper_preset_env_name(variant_name))
    if env_path:
        candidates.append(env_path)
    return candidates


def build_kagglehub_preset_dir_candidates(variant_name):
    candidates = []
    cache_root = Path.home() / ".cache" / "kagglehub" / "models" / "keras"
    variant_root = cache_root / "whisper" / "keras" / variant_name
    if variant_root.exists():
        version_dirs = []
        for candidate in variant_root.iterdir():
            if candidate.is_dir():
                version_dirs.append(candidate)
        version_dirs.sort(key=extract_preset_dir_version, reverse=True)
        candidates.extend(version_dirs)
    return candidates


def require_whisper_preset_dir(variant_name):
    preset_dir = find_whisper_preset_dir(variant_name)
    if preset_dir is None:
        message = build_missing_whisper_preset_message(variant_name)
        raise FileNotFoundError(message)
    return preset_dir


def build_missing_whisper_preset_message(variant_name):
    env_name = build_whisper_preset_env_name(variant_name)
    env_path = os.environ.get(env_name)
    keras_cache_root = Path.home() / ".keras" / "models"
    kaggle_cache_root = (
        Path.home()
        / ".cache"
        / "kagglehub"
        / "models"
        / "keras"
        / "whisper"
        / "keras"
        / variant_name
    )
    candidates = build_preset_dir_candidates(variant_name)
    candidates = [str(Path(path).expanduser()) for path in candidates]
    message = "No local {} preset directory found.".format(variant_name)
    message += " {}={!r}.".format(env_name, env_path)
    message += " Keras cache scan pattern: {}.".format(
        keras_cache_root / "*{}*".format(variant_name)
    )
    message += " Kaggle cache root: {}.".format(kaggle_cache_root)
    if candidates:
        message += " Checked candidates: {}.".format(candidates)
    return message


def build_whisper_preset_env_name(variant_name):
    if variant_name == "whisper_base_en":
        return WHISPER_BASE_EN_PRESET_ENV
    return "PAZ_{}_PRESET_DIR".format(variant_name.upper())


def has_whisper_preset_files(preset_dir):
    config_path = preset_dir / "config.json"
    weights_path = find_preset_weights_path(preset_dir)
    return config_path.exists() and weights_path is not None


def find_preset_weights_path(preset_dir):
    single_file = preset_dir / "model.weights.h5"
    if single_file.exists():
        return single_file
    sharded_config = preset_dir / "model.weights.json"
    if sharded_config.exists():
        return sharded_config
    return None


def extract_preset_dir_version(path):
    try:
        return int(path.name)
    except ValueError:
        return -1


def load_reference_whisper_preset_names():
    module_name = "local_whisper_presets"
    module = load_reference_module(
        module_name, "src/models/whisper/whisper_presets.py"
    )
    return tuple(module.backbone_presets.keys())


def validate_unique_weight_paths(weight_pairs, label):
    duplicate_paths = find_duplicate_weight_paths(weight_pairs)
    if duplicate_paths:
        raise ValueError(
            "Duplicate {} weight paths found: {}".format(
                label, format_weight_paths(duplicate_paths)
            )
        )


def validate_weight_count_alignment(clean_weights, reference_weights):
    if len(clean_weights) == len(reference_weights):
        return
    clean_paths = extract_weight_paths(clean_weights)
    reference_paths = extract_weight_paths(reference_weights)
    clean_only = find_extra_weight_paths(clean_paths, reference_paths)
    reference_only = find_extra_weight_paths(reference_paths, clean_paths)
    raise ValueError(
        "Weight count mismatch: clean={} reference={}. "
        "Clean-only paths: {}. Reference-only paths: {}.".format(
            len(clean_weights),
            len(reference_weights),
            format_weight_paths(clean_only),
            format_weight_paths(reference_only),
        )
    )


def validate_weight_path_alignment(clean_weights, reference_weights):
    mismatch = find_first_weight_path_mismatch(clean_weights, reference_weights)
    if mismatch is None:
        return
    index, clean_path, reference_path = mismatch
    clean_window = build_weight_path_window(clean_weights, index)
    reference_window = build_weight_path_window(reference_weights, index)
    raise ValueError(
        "Weight path mismatch at index {}: clean='{}' reference='{}'. "
        "Clean window: {}. Reference window: {}.".format(
            index,
            clean_path,
            reference_path,
            clean_window,
            reference_window,
        )
    )


def validate_weight_shape_alignment(clean_pair, reference_pair):
    clean_path, clean_weight = clean_pair
    reference_path, reference_weight = reference_pair
    if clean_weight.shape == reference_weight.shape:
        return
    raise ValueError(
        "Weight shape mismatch for '{}': clean={} reference={}. "
        "Reference path='{}'.".format(
            clean_path,
            tuple(clean_weight.shape),
            tuple(reference_weight.shape),
            reference_path,
        )
    )


def find_duplicate_weight_paths(weight_pairs):
    duplicate_paths = []
    seen_paths = set()
    for path, _ in weight_pairs:
        if path in seen_paths and path not in duplicate_paths:
            duplicate_paths.append(path)
        seen_paths.add(path)
    return duplicate_paths


def extract_weight_paths(weight_pairs):
    return [path for path, _ in weight_pairs]


def find_extra_weight_paths(left_paths, right_paths):
    right_paths = set(right_paths)
    extra_paths = []
    for path in left_paths:
        if path not in right_paths:
            extra_paths.append(path)
    return extra_paths


def find_first_weight_path_mismatch(clean_weights, reference_weights):
    for index, pairs in enumerate(zip(clean_weights, reference_weights)):
        clean_path = pairs[0][0]
        reference_path = pairs[1][0]
        if clean_path != reference_path:
            return index, clean_path, reference_path
    return None


def build_weight_path_window(weight_pairs, index):
    start = max(index - 1, 0)
    stop = min(index + 2, len(weight_pairs))
    return extract_weight_paths(weight_pairs[start:stop])


def format_weight_paths(paths, max_paths=8):
    if not paths:
        return "[]"
    if len(paths) <= max_paths:
        return str(paths)
    visible_paths = paths[:max_paths]
    hidden_count = len(paths) - max_paths
    return "{} + {} more".format(visible_paths, hidden_count)


def load_reference_whisper_backbone():
    ensure_reference_modules()
    module = sys.modules["keras_hub.src.models.whisper.whisper_backbone"]
    return module.WhisperBackbone


def load_reference_whisper_audio_converter():
    ensure_reference_audio_converter_modules()
    module = sys.modules["keras_hub.src.models.whisper.whisper_audio_converter"]
    return module.WhisperAudioConverter


def ensure_reference_modules():
    if "keras_hub.src.models.whisper.whisper_backbone" in sys.modules:
        return
    build_package_hierarchy()
    build_backbone_stub()
    load_reference_module(
        "keras_hub.src.api_export",
        "src/api_export.py",
    )
    load_reference_module(
        "keras_hub.src.utils.keras_utils",
        "src/utils/keras_utils.py",
    )
    load_reference_module(
        "keras_hub.src.layers.modeling.position_embedding",
        "src/layers/modeling/position_embedding.py",
    )
    load_reference_module(
        "keras_hub.src.layers.modeling.token_and_position_embedding",
        "src/layers/modeling/token_and_position_embedding.py",
    )
    load_reference_module(
        "keras_hub.src.layers.modeling.transformer_layer_utils",
        "src/layers/modeling/transformer_layer_utils.py",
    )
    load_reference_module(
        "keras_hub.src.layers.modeling.cached_multi_head_attention",
        "src/layers/modeling/cached_multi_head_attention.py",
    )
    load_reference_module(
        "keras_hub.src.layers.modeling.transformer_encoder",
        "src/layers/modeling/transformer_encoder.py",
    )
    load_reference_module(
        "keras_hub.src.layers.modeling.transformer_decoder",
        "src/layers/modeling/transformer_decoder.py",
    )
    load_reference_module(
        "keras_hub.src.models.whisper.whisper_cached_multi_head_attention",
        "src/models/whisper/whisper_cached_multi_head_attention.py",
    )
    load_reference_module(
        "keras_hub.src.models.whisper.whisper_encoder",
        "src/models/whisper/whisper_encoder.py",
    )
    load_reference_module(
        "keras_hub.src.models.whisper.whisper_decoder",
        "src/models/whisper/whisper_decoder.py",
    )
    load_reference_module(
        "keras_hub.src.models.whisper.whisper_backbone",
        "src/models/whisper/whisper_backbone.py",
    )


def ensure_reference_audio_converter_modules():
    if "keras_hub.src.models.whisper.whisper_audio_converter" in sys.modules:
        return
    ensure_reference_modules()
    build_package_hierarchy()
    build_tensor_utils_stub()
    build_preset_utils_stub()
    build_python_utils_stub()
    load_reference_module(
        "keras_hub.src.layers.preprocessing.preprocessing_layer",
        "src/layers/preprocessing/preprocessing_layer.py",
    )
    load_reference_module(
        "keras_hub.src.layers.preprocessing.audio_converter",
        "src/layers/preprocessing/audio_converter.py",
    )
    load_reference_module(
        "keras_hub.src.models.whisper.whisper_audio_converter",
        "src/models/whisper/whisper_audio_converter.py",
    )


def build_package_hierarchy():
    module_names = [
        "keras_hub",
        "keras_hub.src",
        "keras_hub.src.layers",
        "keras_hub.src.layers.preprocessing",
        "keras_hub.src.layers.modeling",
        "keras_hub.src.models",
        "keras_hub.src.models.whisper",
        "keras_hub.src.utils",
    ]
    for module_name in module_names:
        if module_name not in sys.modules:
            module = types.ModuleType(module_name)
            module.__path__ = []
            sys.modules[module_name] = module


def build_tensor_utils_stub():
    module_name = "keras_hub.src.utils.tensor_utils"
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)

    def assert_tf_libs_installed(_):
        return None

    module.assert_tf_libs_installed = assert_tf_libs_installed
    sys.modules[module_name] = module


def build_preset_utils_stub():
    module_name = "keras_hub.src.utils.preset_utils"
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)
    module.builtin_presets = lambda cls: {}
    module.find_subclass = lambda preset, cls, backbone_cls: cls
    module.get_preset_loader = lambda preset: None
    module.get_preset_saver = lambda preset_dir: None
    sys.modules[module_name] = module


def build_python_utils_stub():
    module_name = "keras_hub.src.utils.python_utils"
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)

    class classproperty(property):
        def __get__(self, _, owner_cls):
            return self.fget(owner_cls)

    module.classproperty = classproperty
    sys.modules[module_name] = module


def build_backbone_stub():
    module_name = "keras_hub.src.models.backbone"
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)

    class Backbone(keras.Model):
        def __init__(self, *args, dtype=None, **kwargs):
            super().__init__(*args, **kwargs)
            if dtype is not None:
                try:
                    self.dtype_policy = keras.dtype_policies.get(dtype)
                except AttributeError:
                    self.dtype_policy = keras.DTypePolicy(dtype)

        @property
        def token_embedding(self):
            return getattr(self, "_token_embedding", None)

        @token_embedding.setter
        def token_embedding(self, value):
            self._token_embedding = value

    module.Backbone = Backbone
    sys.modules[module_name] = module


def load_reference_module(module_name, relative_path):
    base_path = os.path.join(
        os.path.dirname(__file__),
        "keras_hub",
    )
    file_path = os.path.join(base_path, relative_path)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
