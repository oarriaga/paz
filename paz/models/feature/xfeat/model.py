from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jp
import keras
from keras import ops
from keras.layers import Input, Conv2D, ReLU, ZeroPadding2D
from keras.layers import AveragePooling2D, BatchNormalization, Lambda
from keras import Model

from paz.models.feature.xfeat import backend

INSTANCE_NORM_EPSILON = 1e-5

Features = namedtuple("Features", ["keypoints", "scores", "descriptors"])


def XFeat(model, top_k=4096, threshold=0.05):
    forward = jax.jit(model)
    core = jax.jit(extract_core, static_argnums=(3, 4, 5, 6))

    def call(image):
        tensor, scale = preprocess(image)
        height, width = tensor.shape[1], tensor.shape[2]
        outputs = core(*forward(tensor), height, width, top_k, threshold)
        return finalize(outputs, scale)

    return call


def extract_core(features, logits, heat, height, width, top_k, threshold):
    features = backend.l2_normalize(features[0], axis=-1)
    heatmap = backend.keypoint_heatmap(logits[0])
    grid, scores = backend.dense_scores(heatmap, heat[0], threshold,
                                        height, width)
    scores, chosen = jax.lax.top_k(scores, top_k)
    positions = grid[chosen]
    descriptors = backend.sample_features(features, positions, height,
                                          width, "bicubic")
    return positions, scores, backend.l2_normalize(descriptors, axis=-1)


def finalize(outputs, scale):
    positions, scores, descriptors = (np.asarray(x) for x in outputs)
    valid = scores > 0
    keypoints = positions[valid] * np.asarray(scale)
    return Features(keypoints, scores[valid], descriptors[valid])


def preprocess(image):
    image = jp.asarray(image, jp.float32)
    if image.ndim == 3:
        image = image[None]
    height, width = image.shape[1], image.shape[2]
    target = (height // 32) * 32, (width // 32) * 32
    if target != (height, width):
        image = ops.image.resize(image, target, antialias=False)
    scale = jp.array([width / target[1], height / target[0]])
    return image, scale


def XFeatModel(name="xfeat"):
    image = Input((None, None, 3))
    gray = Lambda(to_grayscale, output_shape=(None, None, 1))(image)
    normed = Lambda(instance_normalize, output_shape=(None, None, 1))(gray)

    channels1 = [(4, 3, 1), (8, 3, 2), (8, 3, 1), (24, 3, 2)]
    x1 = build_block(normed, channels1, "block1")
    skip = build_skip(normed)
    x2 = build_block(add(x1, skip), [(24, 3, 1), (24, 3, 1)], "block2")
    x3 = build_block(x2, [(64, 3, 2), (64, 3, 1), (64, 1, 1)], "block3")
    x4 = build_block(x3, [(64, 3, 2), (64, 3, 1), (64, 3, 1)], "block4")
    channels5 = [(128, 3, 2), (128, 3, 1), (128, 3, 1), (64, 1, 1)]
    x5 = build_block(x4, channels5, "block5")

    x4 = resize_to(x3, x4)
    x5 = resize_to(x3, x5)
    fused = add(add(x3, x4), x5)
    features = build_fusion(fused)

    heatmap = build_heatmap_head(features)
    keypoints = build_keypoint_head(normed)
    return Model(image, [features, keypoints, heatmap], name=name)


def to_grayscale(image):
    return ops.mean(image, axis=-1, keepdims=True)


def instance_normalize(x):
    mean = ops.mean(x, axis=(1, 2), keepdims=True)
    variance = ops.var(x, axis=(1, 2), keepdims=True)
    return (x - mean) / ops.sqrt(variance + INSTANCE_NORM_EPSILON)


def add(a, b):
    return keras.layers.Add()([a, b])


def resize_to(target, source):
    def call(inputs):
        target, source = inputs
        size = ops.shape(target)[1:3]
        return ops.image.resize(source, size, interpolation="bilinear")

    return Lambda(call, output_shape=(None, None, 64))([target, source])


def build_block(x, specs, prefix):
    for index, (filters, kernel, stride) in enumerate(specs):
        x = basic_layer(x, filters, kernel, stride, f"{prefix}_{index}")
    return x


def basic_layer(x, filters, kernel, stride, name):
    x = convolve(x, filters, kernel, stride, False, f"{name}_conv")
    x = BatchNormalization(center=False, scale=False,
                           epsilon=1e-5, name=f"{name}_bn")(x)
    return ReLU()(x)


def convolve(x, filters, kernel, stride, use_bias, name):
    if kernel == 3:
        x = ZeroPadding2D(1)(x)
    return Conv2D(filters, kernel, strides=stride, padding="valid",
                  use_bias=use_bias, name=name)(x)


def build_skip(x):
    x = AveragePooling2D(4, strides=4)(x)
    return Conv2D(24, 1, use_bias=True, name="skip_conv")(x)


def build_fusion(x):
    x = basic_layer(x, 64, 3, 1, "fusion_0")
    x = basic_layer(x, 64, 3, 1, "fusion_1")
    return Conv2D(64, 1, use_bias=True, name="fusion_out")(x)


def build_heatmap_head(x):
    x = basic_layer(x, 64, 1, 1, "heatmap_0")
    x = basic_layer(x, 64, 1, 1, "heatmap_1")
    x = Conv2D(1, 1, use_bias=True, name="heatmap_out")(x)
    return keras.activations.sigmoid(x)


def build_keypoint_head(normed):
    x = Lambda(unfold_grid, output_shape=(None, None, 64))(normed)
    x = basic_layer(x, 64, 1, 1, "keypoint_0")
    x = basic_layer(x, 64, 1, 1, "keypoint_1")
    x = basic_layer(x, 64, 1, 1, "keypoint_2")
    return Conv2D(65, 1, use_bias=True, name="keypoint_out")(x)


def unfold_grid(x, window=8):
    batch = ops.shape(x)[0]
    height, width = ops.shape(x)[1], ops.shape(x)[2]
    rows, columns = height // window, width // window
    x = ops.reshape(x, (batch, rows, window, columns, window, 1))
    x = ops.transpose(x, (0, 1, 3, 5, 2, 4))
    return ops.reshape(x, (batch, rows, columns, window * window))
