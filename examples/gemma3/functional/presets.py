from examples.gemma3.functional.keras_hub_utils import ensure_keras_hub


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
    output = {}
    output["vocabulary_size"] = config["vocabulary_size"]
    output["image_size"] = config["image_size"]
    output["num_layers"] = config["num_layers"]
    output["num_query_heads"] = config["num_query_heads"]
    output["num_key_value_heads"] = config["num_key_value_heads"]
    output["hidden_dim"] = config["hidden_dim"]
    output["intermediate_dim"] = config["intermediate_dim"]
    output["head_dim"] = config["head_dim"]
    query_norm = config.get("query_head_dim_normalize", True)
    output["query_head_dim_normalize"] = query_norm
    output["use_query_key_norm"] = config.get("use_query_key_norm", True)
    output["use_post_ffw_norm"] = config.get("use_post_ffw_norm", False)
    post_attention = config.get("use_post_attention_norm", False)
    output["use_post_attention_norm"] = post_attention
    output["attention_logit_soft_cap"] = config.get("attention_logit_soft_cap")
    output["final_logit_soft_cap"] = config.get("final_logit_soft_cap")
    use_sliding = config.get("use_sliding_window_attention", False)
    output["use_sliding_window_attention"] = use_sliding
    output["sliding_window_size"] = config.get("sliding_window_size", 1024)
    local_scale = config.get("local_rope_scaling_factor", 1.0)
    global_scale = config.get("global_rope_scaling_factor", 1.0)
    use_bidirectional = config.get("use_bidirectional_attention", False)
    output["local_rope_scaling_factor"] = local_scale
    output["global_rope_scaling_factor"] = global_scale
    output["use_bidirectional_attention"] = use_bidirectional
    output["layer_norm_epsilon"] = config.get("layer_norm_epsilon", 1e-6)
    output["dropout"] = config.get("dropout", 0.0)
    return output


def _parse_vision_config(vision_encoder):
    if vision_encoder is None:
        return None
    config = vision_encoder["config"]
    output = {}
    output["image_size"] = config["image_size"]
    output["patch_size"] = config["patch_size"]
    output["num_heads"] = config["num_heads"]
    output["hidden_dim"] = config["hidden_dim"]
    output["num_layers"] = config["num_layers"]
    output["intermediate_dim"] = config["intermediate_dim"]
    output["output_dim"] = config["output_dim"]
    output["pool_size"] = config.get("pool_size", 14)
    output["layer_norm_epsilon"] = config.get("layer_norm_epsilon", 1e-6)
    output["dropout"] = config.get("dropout", 0.0)
    return output


def load_gemma3_preset_config(preset):
    config = _load_preset_config(preset)
    backbone_config = _parse_backbone_config(config["config"])
    vision_config = _parse_vision_config(config["config"].get("vision_encoder"))
    metadata = get_gemma3_preset_metadata(str(preset))
    result = {}
    result["backbone_config"] = backbone_config
    result["vision_config"] = vision_config
    result["metadata"] = metadata
    return result
