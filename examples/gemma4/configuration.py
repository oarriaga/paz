import json

from .model import TextBackboneArgs

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
