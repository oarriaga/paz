import json
from collections import namedtuple

TEXT_BACKBONE_FIELDS = (
    "vocabulary_size image_size num_layers num_query_heads "
    "num_key_value_heads hidden_dim intermediate_dim head_dim "
    "attention_logit_soft_cap final_logit_soft_cap "
    "use_sliding_window_attention sliding_window_size "
    "sliding_window_pattern global_head_dim "
    "local_rope_wavelength global_rope_wavelength "
    "local_rope_scaling_factor global_rope_scaling_factor "
    "global_rope_partial_rotary_factor "
    "use_bidirectional_attention layer_norm_epsilon dropout dtype "
    "hidden_size_per_layer_input num_kv_shared_layers global_layer_indices "
    "use_double_wide_mlp"
)
TextBackboneArgs = namedtuple("TextBackboneArgs", TEXT_BACKBONE_FIELDS)

CONFIGS = {
    "gemma4_2b": {
        "vocabulary_size": 262_144,
        "image_size": 896,
        "num_layers": 35,
        "num_query_heads": 8,
        "num_key_value_heads": 1,
        "hidden_dim": 1536,
        "intermediate_dim": 6144,
        "head_dim": 256,
        "attention_logit_soft_cap": None,
        "final_logit_soft_cap": 30.0,
        "use_sliding_window_attention": True,
        "sliding_window_size": 512,
        "sliding_window_pattern": 5,
        "global_head_dim": 512,
        "local_rope_wavelength": 10_000.0,
        "global_rope_wavelength": 1_000_000.0,
        "local_rope_scaling_factor": 1.0,
        "global_rope_scaling_factor": 1.0,
        "global_rope_partial_rotary_factor": 0.25,
        "use_bidirectional_attention": False,
        "layer_norm_epsilon": 1e-6,
        "dropout": 0.0,
        "dtype": "bfloat16",
        "hidden_size_per_layer_input": 256,
        "num_kv_shared_layers": 20,
        "global_layer_indices": None,
        "use_double_wide_mlp": True,
    },
    "gemma4_4b": {
        "vocabulary_size": 262_144,
        "image_size": 896,
        "num_layers": 42,
        "num_query_heads": 8,
        "num_key_value_heads": 2,
        "hidden_dim": 2560,
        "intermediate_dim": 10240,
        "head_dim": 256,
        "attention_logit_soft_cap": None,
        "final_logit_soft_cap": 30.0,
        "use_sliding_window_attention": True,
        "sliding_window_size": 512,
        "sliding_window_pattern": 6,
        "global_head_dim": 512,
        "local_rope_wavelength": 10_000.0,
        "global_rope_wavelength": 1_000_000.0,
        "local_rope_scaling_factor": 1.0,
        "global_rope_scaling_factor": 1.0,
        "global_rope_partial_rotary_factor": 0.25,
        "use_bidirectional_attention": False,
        "layer_norm_epsilon": 1e-6,
        "dropout": 0.0,
        "dtype": "bfloat16",
        "hidden_size_per_layer_input": 256,
        "num_kv_shared_layers": 18,
        "global_layer_indices": None,
        "use_double_wide_mlp": False,
    },
}

# Defaults for fields added after some configs were serialized, so older
# config.json files still load.
LEGACY_DEFAULTS = {
    "local_rope_wavelength": 10_000.0,
    "global_rope_wavelength": 1_000_000.0,
    # Only the E2B config predates this field, and E2B uses double-wide MLPs.
    "use_double_wide_mlp": True,
}


def to_backbone_args(model_name):
    return TextBackboneArgs(**CONFIGS[model_name])


def save_config(config, path):
    with open(str(path), "w", encoding="utf-8") as file:
        json.dump(config._asdict(), file, indent=2)


def load_config(path):
    with open(str(path), encoding="utf-8") as file:
        values = json.load(file)
    values = {**LEGACY_DEFAULTS, **values}
    values["global_layer_indices"] = build_global_indices(values)
    return TextBackboneArgs(**values)


def build_global_indices(values):
    indices = values.get("global_layer_indices")
    if indices is None:
        return None
    return tuple(indices)


def build_cache_head_dim(config):
    if config.global_head_dim is not None:
        return config.global_head_dim
    return config.head_dim


def is_global_attention_layer(config, layer_index):
    if config.global_layer_indices is not None:
        return layer_index in config.global_layer_indices
    pattern_index = layer_index % config.sliding_window_pattern
    return pattern_index == config.sliding_window_pattern - 1


def is_kv_shared_layer(config, layer_index):
    if not config.num_kv_shared_layers:
        return False
    return layer_index >= config.num_layers - config.num_kv_shared_layers


def layer_attention_type(config, layer_index):
    if is_global_attention_layer(config, layer_index):
        return "global"
    return "local"


def build_kv_source_map(config):
    """Map each kv_shared layer to the most recent layer of the same type.

    Returns a dict {layer_index: source_index} for kv_shared layers only,
    mirroring the keras-hub backbone precomputation of _kv_source.
    """
    num_shared = config.num_kv_shared_layers
    if not num_shared:
        return {}
    first_shared = config.num_layers - num_shared
    non_shared_types = []
    for layer_index in range(first_shared):
        non_shared_types.append(layer_attention_type(config, layer_index))
    kv_source = {}
    for layer_index in range(first_shared, config.num_layers):
        layer_type = layer_attention_type(config, layer_index)
        source_index = last_matching_layer(non_shared_types, layer_type)
        if source_index is not None:
            kv_source[layer_index] = source_index
    return kv_source


def last_matching_layer(layer_types, layer_type):
    for source_index in range(len(layer_types) - 1, -1, -1):
        if layer_types[source_index] == layer_type:
            return source_index
    return None


def build_feedforward_dim(config, layer_index):
    if config.use_double_wide_mlp and is_kv_shared_layer(config, layer_index):
        return config.intermediate_dim * 2
    return config.intermediate_dim


def build_head_dim(config, is_global):
    if is_global and config.global_head_dim is not None:
        return config.global_head_dim
    return config.head_dim


def use_sliding_window(config, is_global):
    return config.use_sliding_window_attention and not is_global


def build_rope_wavelength(config, is_global):
    if is_global:
        return config.global_rope_wavelength
    return config.local_rope_wavelength


def build_rope_scaling_factor(config, is_global):
    if is_global:
        return config.global_rope_scaling_factor
    return config.local_rope_scaling_factor


def build_partial_rotary_factor(config, is_global):
    if is_global:
        return config.global_rope_partial_rotary_factor
    return 1.0
