"""SAM 2 prompt encoder: point, box, and mask embeddings.

Points and box corners share a random-Fourier positional encoding and add a
learned per-label embedding. Coordinates arrive in the resized 1024 space.
The dense positional encoding and the no-mask embedding are computed eagerly
from the loaded weights and fed to the mask decoder as constants.
"""
import math

import keras
from keras import Input, Model, ops
from keras.layers import Conv2D, LayerNormalization, Layer

from paz.models.foundation.sam2.configuration import IMAGE_SIZE
from paz.models.foundation.sam2.configuration import PROMPT_EMBED_DIM
from paz.models.foundation.sam2.configuration import BACKBONE_STRIDE
from paz.models.foundation.sam2.layers import ChannelBias

GRID = IMAGE_SIZE // BACKBONE_STRIDE
MASK_INPUT = 4 * GRID


@keras.saving.register_keras_serializable(package="paz")
class RandomFourierEmbedding(Layer):
    def __init__(self, num_features, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features

    def build(self, input_shape):
        shape = (2, self.num_features)
        kwargs = dict(name="matrix", shape=shape, initializer="zeros")
        self.matrix = self.add_weight(trainable=False, **kwargs)

    def call(self, coordinates):
        coordinates = 2.0 * coordinates - 1.0
        projected = ops.matmul(coordinates, self.matrix)
        projected = 2.0 * math.pi * projected
        return ops.concatenate([ops.sin(projected), ops.cos(projected)], -1)

    def get_config(self):
        arguments = {"num_features": self.num_features}
        return {**super().get_config(), **arguments}


@keras.saving.register_keras_serializable(package="paz")
class PointLabelEmbedding(Layer):
    def build(self, input_shape):
        corners = dict(name="corners", shape=(4, PROMPT_EMBED_DIM))
        point = dict(name="not_a_point", shape=(1, PROMPT_EMBED_DIM))
        self.corners = self.add_weight(initializer="zeros", **corners)
        self.not_a_point = self.add_weight(initializer="zeros", **point)

    def call(self, inputs):
        encoding, labels = inputs
        labels = labels[..., None]
        encoding = ops.where(labels == -1, self.not_a_point[None], encoding)
        for label in range(4):
            corner = self.corners[label]
            encoding = ops.where(labels == label, encoding + corner, encoding)
        return encoding


def build_points(name="sam2_prompt_encoder"):
    coordinates = Input((None, 2), name="point_coords")
    labels = Input((None,), name="point_labels")
    normalized = (coordinates + 0.5) / IMAGE_SIZE
    layer = RandomFourierEmbedding(PROMPT_EMBED_DIM // 2, name="prompt_pe")
    encoding = layer(normalized)
    sparse = PointLabelEmbedding(name="point_label_embed")([encoding, labels])
    return Model((coordinates, labels), sparse, name=name)


def build_mask_downscaling(name="sam2_mask_downscaling"):
    masks = Input((MASK_INPUT, MASK_INPUT, 1), name="mask_input")
    x = Conv2D(4, 2, strides=2, name="mask_down_0")(masks)
    x = LayerNormalization(axis=-1, epsilon=1e-6, name="mask_down_ln0")(x)
    x = ops.gelu(x, approximate=False)
    x = Conv2D(16, 2, strides=2, name="mask_down_3")(x)
    x = LayerNormalization(axis=-1, epsilon=1e-6, name="mask_down_ln3")(x)
    x = ops.gelu(x, approximate=False)
    dense = Conv2D(PROMPT_EMBED_DIM, 1, name="mask_down_6")(x)
    empty = ops.zeros_like(dense)
    no_mask = ChannelBias(PROMPT_EMBED_DIM, name="no_mask_embed")(empty)
    return Model(masks, (dense, no_mask), name=name)


def dense_positional_encoding(point_model):
    layer = point_model.get_layer("prompt_pe")
    axis = (ops.arange(GRID, dtype="float32") + 0.5) / GRID
    grid_y, grid_x = ops.meshgrid(axis, axis, indexing="ij")
    coordinates = ops.stack([grid_x, grid_y], axis=-1)
    return layer(coordinates)
