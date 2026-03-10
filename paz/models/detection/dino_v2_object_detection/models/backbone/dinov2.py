import json
import math
import os

import keras
from keras import layers, ops, Model

from paz.models.detection.dino_v2_object_detection.models.backbone.dinov2_with_windowed_attn import (
    WindowedDinov2Model,
    dinov2_windowed_small,
    dinov2_windowed_base,
    dinov2_windowed_large,
    dinov2_windowed_giant,
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


@keras.saving.register_keras_serializable(package="backbone")
class DinoV2(layers.Layer):
    """
    Keras3 port of PyTorch DinoV2 backbone wrapper.
    Creates a WindowedDinov2Model, handles windowed attention setup,
    and returns multi-scale feature maps as a list of (B, H, W, C) tensors.
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
        self.window_block_indexes = window_block_indexes

        # Load config from JSON
        dino_config = get_config(size, use_registers)

        # Set image_size to the actual target resolution so that position
        # embeddings match the runtime patch-grid size without interpolation.
        target_resolution = shape[0]
        if target_resolution != dino_config["image_size"]:
            dino_config["image_size"] = target_resolution

        if patch_size != 14:
            dino_config["patch_size"] = patch_size

        num_register_tokens = (
            dino_config.get("num_register_tokens", 4) if use_registers else 0
        )

        num_hidden_layers = dino_config.get("num_hidden_layers", 12)

        if window_block_indexes is None:
            pt_out_indices = [idx + 1 for idx in out_feature_indexes]
            wb_indexes = set(range(num_hidden_layers + 1))
            wb_indexes.difference_update(pt_out_indices)
            window_block_indexes = sorted(list(wb_indexes))

        self.window_block_indexes = window_block_indexes

        self.encoder = WindowedDinov2Model(
            image_size=dino_config["image_size"],
            patch_size=dino_config.get("patch_size", 14),
            num_channels=dino_config.get("num_channels", 3),
            hidden_size=dino_config["hidden_size"],
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=dino_config["num_attention_heads"],
            mlp_ratio=dino_config.get("mlp_ratio", 4),
            drop_path_rate=drop_path_rate,
            layer_norm_eps=dino_config.get("layer_norm_eps", 1e-6),
            use_swiglu_ffn=dino_config.get("use_swiglu_ffn", False),
            num_register_tokens=num_register_tokens,
            num_windows=num_windows,
            window_block_indexes=window_block_indexes,
            init_values=init_values,
            out_feature_indexes=out_feature_indexes,
            interpolate_antialias=dino_config.get(
                "interpolate_antialias", interpolate_antialias
            ),
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
        """Extract multi-scale feature maps from the encoder.

        Args:
            x (Tensor): Input tensor of shape (B, H, W, C).
            training (bool): Whether in training mode.

        Returns:
            list: Feature tensors, each of shape (B, H', W', embed_dim),
                one per entry in out_feature_indexes.
        """
        batch_size = ops.shape(x)[0]
        height = ops.shape(x)[1]
        width = ops.shape(x)[2]
        patch_h = height // self.patch_size
        patch_w = width // self.patch_size
        embed_dim = self.encoder.hidden_size
        num_register_tokens = self.encoder.num_register_tokens
        nw = self.num_windows

        embedding_output = self.encoder.embeddings(x, training=training)

        hidden_states = embedding_output
        outputs = []
        for i, layer_module in enumerate(self.encoder.encoder.encoder_layers):
            run_full_attention = i not in self.encoder.encoder.window_block_indexes
            hidden_states = layer_module(
                hidden_states,
                training=training,
                run_full_attention=run_full_attention,
            )

            if i in self.out_feature_indexes:
                normed = self.encoder.layernorm(hidden_states)

                start_idx = 1 + num_register_tokens
                feature = normed[:, start_idx:]

                if nw > 1:
                    pad_h = (nw - patch_h % nw) % nw
                    pad_w = (nw - patch_w % nw) % nw
                    padded_h = patch_h + pad_h
                    padded_w = patch_w + pad_w
                    h_w = padded_h // nw
                    w_w = padded_w // nw

                    feature = ops.reshape(
                        feature, (batch_size, nw, nw, h_w, w_w, embed_dim)
                    )
                    feature = ops.transpose(feature, (0, 1, 3, 2, 4, 5))
                    feature = ops.reshape(
                        feature, (batch_size, padded_h, padded_w, embed_dim)
                    )

                    if pad_h > 0 or pad_w > 0:
                        feature = feature[:, :patch_h, :patch_w, :]
                else:
                    feature = ops.reshape(
                        feature, (batch_size, patch_h, patch_w, embed_dim)
                    )

                outputs.append(feature)

        return outputs
