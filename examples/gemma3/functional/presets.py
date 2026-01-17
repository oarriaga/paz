from collections import namedtuple

from examples.gemma3.functional.keras_hub_utils import ensure_keras_hub
from examples.gemma3.functional.model import build_backbone_config
from examples.gemma3.functional.vision import build_vision_encoder_config


Gemma3PresetConfig = namedtuple(
    "Gemma3PresetConfig", ["backbone_config", "vision_config", "metadata"]
)


def list_gemma3_presets():
    ensure_keras_hub()
    from keras_hub.src.models.gemma3.gemma3_presets import backbone_presets

    return sorted(backbone_presets.keys())


def get_gemma3_preset_metadata(preset):
    ensure_keras_hub()
    from keras_hub.src.models.gemma3.gemma3_presets import backbone_presets

    preset = str(preset)
    if preset in backbone_presets:
        return backbone_presets[preset].get("metadata", {})
    return {}


def _load_preset_config(preset):
    ensure_keras_hub()
    from keras_hub.src.utils.preset_utils import CONFIG_FILE
    from keras_hub.src.utils.preset_utils import load_json

    return load_json(str(preset), CONFIG_FILE)


def _parse_backbone_config(config):
    return build_backbone_config(
        vocabulary_size=config["vocabulary_size"],
        image_size=config["image_size"],
        num_layers=config["num_layers"],
        num_query_heads=config["num_query_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        hidden_dim=config["hidden_dim"],
        intermediate_dim=config["intermediate_dim"],
        head_dim=config["head_dim"],
        query_head_dim_normalize=config.get("query_head_dim_normalize", True),
        use_query_key_norm=config.get("use_query_key_norm", True),
        use_post_ffw_norm=config.get("use_post_ffw_norm", False),
        use_post_attention_norm=config.get("use_post_attention_norm", False),
        attention_logit_soft_cap=config.get("attention_logit_soft_cap"),
        final_logit_soft_cap=config.get("final_logit_soft_cap"),
        use_sliding_window_attention=config.get(
            "use_sliding_window_attention", False
        ),
        sliding_window_size=config.get("sliding_window_size", 1024),
        local_rope_scaling_factor=config.get("local_rope_scaling_factor", 1.0),
        global_rope_scaling_factor=config.get(
            "global_rope_scaling_factor", 1.0
        ),
        use_bidirectional_attention=config.get("use_bidirectional_attention", False),
        layer_norm_epsilon=config.get("layer_norm_epsilon", 1e-6),
        dropout=config.get("dropout", 0.0),
    )


def _parse_vision_config(vision_encoder):
    if vision_encoder is None:
        return None
    config = vision_encoder["config"]
    return build_vision_encoder_config(
        image_size=config["image_size"],
        patch_size=config["patch_size"],
        num_heads=config["num_heads"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        intermediate_dim=config["intermediate_dim"],
        output_dim=config["output_dim"],
        pool_size=config.get("pool_size", 14),
        layer_norm_epsilon=config.get("layer_norm_epsilon", 1e-6),
        dropout=config.get("dropout", 0.0),
    )


def load_gemma3_preset_config(preset):
    config = _load_preset_config(preset)
    backbone_config = _parse_backbone_config(config["config"])
    vision_config = _parse_vision_config(config["config"].get("vision_encoder"))
    metadata = get_gemma3_preset_metadata(str(preset))
    return Gemma3PresetConfig(backbone_config, vision_config, metadata)

