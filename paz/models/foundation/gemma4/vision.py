from collections import namedtuple
from pathlib import Path

from keras import Model
from keras.layers import Input

from paz.models.foundation.gemma4.layers.vision import (
    build_average_pooling, build_patch_embedder, build_real_patch_mask,
    build_vision_attention_mask, build_vision_output, vision_decoder_block)

VISION_ENCODER_NAME = "gemma4_vision_encoder"
VISION_ENCODER_FIELDS = (
    "image_size patch_size pool_size hidden_dim num_layers num_heads "
    "num_key_value_heads head_dim intermediate_dim output_dim "
    "position_embedding_size rope_wavelength layer_norm_epsilon dropout "
    "max_patches dtype"
)
VisionEncoderArgs = namedtuple("VisionEncoderArgs", VISION_ENCODER_FIELDS)


def build_vision_encoder_args(**overrides):
    values = {
        "image_size": 24,
        "patch_size": 4,
        "pool_size": 3,
        "hidden_dim": 16,
        "num_layers": 2,
        "num_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "intermediate_dim": 32,
        "output_dim": 16,
        "position_embedding_size": 32,
        "rope_wavelength": 100.0,
        "layer_norm_epsilon": 1e-6,
        "dropout": 0.0,
        "dtype": "float32",
    }
    values.update(overrides)
    side = values["image_size"] // values["patch_size"]
    values.setdefault("max_patches", side * side)
    return VisionEncoderArgs(**values)


def num_patches(config):
    return config.max_patches


def build_vision_encoder(config, weights_path=None, name=VISION_ENCODER_NAME):
    patch_dim = 3 * config.patch_size ** 2
    pixel_values = Input((num_patches(config), patch_dim), name="pixel_values")
    pixel_position_ids = Input(
        (num_patches(config), 2), dtype="int32", name="pixel_position_ids")
    hidden = build_patch_embedder(pixel_values, pixel_position_ids, config)
    real_mask = build_real_patch_mask(pixel_position_ids)
    key_mask = build_vision_attention_mask(pixel_position_ids)
    for layer_index in range(config.num_layers):
        block_name = "encoder_block_{}".format(layer_index)
        hidden = vision_decoder_block(
            hidden, key_mask, pixel_position_ids, config, block_name)
        hidden = hidden * real_mask
    pooled = build_average_pooling(hidden, pixel_position_ids, config)
    output = build_vision_output(pooled, config)
    inputs = {"pixel_values": pixel_values,
              "pixel_position_ids": pixel_position_ids}
    model = Model(inputs, output, name=name)
    if weights_path is not None:
        model.load_weights(str(Path(weights_path)))
    return model
