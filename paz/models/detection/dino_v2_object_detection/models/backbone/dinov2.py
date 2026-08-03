import json
import os

from paz.models.foundation.dinov2.models.windowed_vision_transformer import (
    WindowedDinov2Model,
)


SIZE_TO_WIDTH = {
    "tiny": 192,
    "small": 384,
    "base": 768,
    "large": 1024,
}

SIZE_TO_CONFIG = {
    "small": "dinov2_small.json",
    "base": "dinov2_base.json",
    "large": "dinov2_large.json",
}

SIZE_TO_CONFIG_WITH_REGISTERS = {
    "small": "dinov2_with_registers_small.json",
    "base": "dinov2_with_registers_base.json",
    "large": "dinov2_with_registers_large.json",
}


def load_dino_config(size, use_registers):
    names = SIZE_TO_CONFIG_WITH_REGISTERS if use_registers else SIZE_TO_CONFIG
    current_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(current_dir, "dinov2_configs")
    config_path = os.path.join(configs_dir, names[size])
    with open(config_path, "r") as config_file:
        return json.load(config_file)


def resolve_window_block_indexes(window_block_indexes, out_feature_indexes, depth):  # fmt: skip
    if window_block_indexes is None:
        pt_out_indices = [index + 1 for index in out_feature_indexes]
        indexes = set(range(depth + 1))
        indexes.difference_update(pt_out_indices)
        window_block_indexes = sorted(indexes)
    return window_block_indexes


def resolve_dino_config(size, use_registers, shape, patch_size):
    config = load_dino_config(size, use_registers)
    if shape[0] != config["image_size"]:
        config["image_size"] = shape[0]
    if patch_size != 14:
        config["patch_size"] = patch_size
    return config


def read_register_tokens(config, use_registers):
    tokens = 0
    if use_registers:
        tokens = config.get("num_register_tokens", 4)
    return tokens


def annotate_encoder(model, size, use_registers, config, out_feature_indexes):
    model.hidden_size = config["hidden_size"]
    model.num_hidden_layers = config.get("num_hidden_layers", 12)
    model.size = size
    model.use_registers = use_registers
    width = SIZE_TO_WIDTH[size]
    model._out_feature_channels = [width] * len(out_feature_indexes)
    return model


# use_windowed_attn and positional_encoding_size are unused here but stay in
# the signature: Backbone and the backbone tests pass them by keyword.
def DinoV2(shape=(640, 640), out_feature_indexes=None, size="base", use_registers=True, use_windowed_attn=True, patch_size=14, num_windows=4, window_block_indexes=None, positional_encoding_size=37, drop_path_rate=0.0, init_values=1e-5, name="dinov2_encoder"):  # fmt: skip
    if out_feature_indexes is None:
        out_feature_indexes = [2, 4, 5, 9]
    config = resolve_dino_config(size, use_registers, shape, patch_size)
    depth = config.get("num_hidden_layers", 12)
    windows = resolve_window_block_indexes(window_block_indexes, out_feature_indexes, depth)  # fmt: skip
    keys = ("image_size", "patch_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "mlp_ratio", "use_swiglu_ffn", "num_register_tokens", "num_windows", "window_block_indexes", "init_values", "drop_path_rate", "out_feature_indexes", "name")  # fmt: skip
    values = (config["image_size"], config.get("patch_size", 14), config["hidden_size"], depth, config["num_attention_heads"], config.get("mlp_ratio", 4), config.get("use_swiglu_ffn", False), read_register_tokens(config, use_registers), num_windows, windows, init_values, drop_path_rate, out_feature_indexes, name)  # fmt: skip
    model = WindowedDinov2Model(**dict(zip(keys, values)))
    args = (size, use_registers, config, out_feature_indexes)
    return annotate_encoder(model, *args)
