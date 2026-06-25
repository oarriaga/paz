import json
import os

import keras
from keras import layers

from paz.models.foundation.dinov2.models.windowed_vision_transformer import (
    WindowedDinov2Model,
)


size_to_width = {
    "tiny": 192,
    "small": 384,
    "base": 768,
    "large": 1024,
}

size_to_config = {
    "small": "dinov2_small.json",
    "base": "dinov2_base.json",
    "large": "dinov2_large.json",
}

size_to_config_with_registers = {
    "small": "dinov2_with_registers_small.json",
    "base": "dinov2_with_registers_base.json",
    "large": "dinov2_with_registers_large.json",
}


def get_config(size, use_registers):
    config_dict = size_to_config_with_registers if use_registers else size_to_config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(current_dir, "dinov2_configs")
    config_path = os.path.join(configs_dir, config_dict[size])
    with open(config_path, "r") as f:
        dino_config = json.load(f)
    return dino_config


def resolve_window_block_indexes(window_block_indexes, out_feature_indexes, depth):
    if window_block_indexes is not None:
        return window_block_indexes
    pt_out_indices = [idx + 1 for idx in out_feature_indexes]
    indexes = set(range(depth + 1))
    indexes.difference_update(pt_out_indices)
    return sorted(indexes)


@keras.saving.register_keras_serializable(package="backbone")
class DinoV2(layers.Layer):
    """Keras3 port of the PyTorch DinoV2 backbone wrapper.

    Builds a functional windowed DINOv2 feature model and returns
    multi-scale feature maps as a list of (B, H, W, C) tensors.
    """

    def __init__(
        self,
        shape=(640, 640),
        out_feature_indexes=None,
        size="base",
        use_registers=True,
        use_windowed_attn=True,
        patch_size=14,
        num_windows=4,
        window_block_indexes=None,
        positional_encoding_size=37,
        drop_path_rate=0.0,
        init_values=1e-5,
        interpolate_antialias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if out_feature_indexes is None:
            out_feature_indexes = [2, 4, 5, 9]

        self.shape = shape
        self.patch_size = patch_size
        self.num_windows = num_windows
        self.out_feature_indexes = out_feature_indexes
        self.size = size
        self.use_registers = use_registers

        dino_config = get_config(size, use_registers)
        if shape[0] != dino_config["image_size"]:
            dino_config["image_size"] = shape[0]
        if patch_size != 14:
            dino_config["patch_size"] = patch_size

        num_register_tokens = (
            dino_config.get("num_register_tokens", 4) if use_registers else 0
        )
        num_hidden_layers = dino_config.get("num_hidden_layers", 12)
        window_block_indexes = resolve_window_block_indexes(
            window_block_indexes, out_feature_indexes, num_hidden_layers
        )
        self.window_block_indexes = window_block_indexes

        use_swiglu = dino_config.get("use_swiglu_ffn", False)
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = dino_config["hidden_size"]
        self.num_register_tokens = num_register_tokens
        self.use_swiglu = use_swiglu

        self.feature_model = WindowedDinov2Model(
            image_size=dino_config["image_size"],
            patch_size=dino_config.get("patch_size", 14),
            hidden_size=dino_config["hidden_size"],
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=dino_config["num_attention_heads"],
            mlp_ratio=dino_config.get("mlp_ratio", 4),
            use_swiglu_ffn=use_swiglu,
            num_register_tokens=num_register_tokens,
            num_windows=num_windows,
            window_block_indexes=window_block_indexes,
            init_values=init_values,
            drop_path_rate=drop_path_rate,
            out_feature_indexes=out_feature_indexes,
            name="dinov2_encoder",
        )

        self._out_feature_channels = [size_to_width[size]] * len(out_feature_indexes)
        self._export = False
        self._drop_path_rate = drop_path_rate
        self._init_values = init_values
        self._positional_encoding_size = positional_encoding_size
        self._interpolate_antialias = interpolate_antialias
        self._use_windowed_attn = use_windowed_attn

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "shape": self.shape,
                "out_feature_indexes": self.out_feature_indexes,
                "size": self.size,
                "use_registers": self.use_registers,
                "use_windowed_attn": self._use_windowed_attn,
                "patch_size": self.patch_size,
                "num_windows": self.num_windows,
                "window_block_indexes": self.window_block_indexes,
                "positional_encoding_size": self._positional_encoding_size,
                "drop_path_rate": self._drop_path_rate,
                "init_values": self._init_values,
                "interpolate_antialias": self._interpolate_antialias,
            }
        )
        return config

    def call(self, x, training=None):
        """Extract multi-scale feature maps from the functional model.

        Args:
            x (Tensor): Input tensor of shape (B, H, W, C).
            training (bool): Whether in training mode.

        Returns:
            list: Feature tensors, each of shape (B, H', W', embed_dim),
                one per entry in out_feature_indexes.
        """
        return self.feature_model(x, training=training)
