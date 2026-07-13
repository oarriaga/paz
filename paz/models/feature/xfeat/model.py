from collections import namedtuple

import jax
import jax.numpy as jp
from keras import ops
from keras.layers import Input, Conv2D, ReLU, ZeroPadding2D
from keras.layers import AveragePooling2D, BatchNormalization, UpSampling2D
from keras import Model
from keras.utils import get_file

from paz.backend import features
from paz.models.feature.xfeat import backend

INSTANCE_NORM_EPSILON = 1e-5
WEIGHTS_URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.25/xfeat_paz_jax.weights.h5"  # fmt: skip

Features = namedtuple("Features", ["keypoints", "scores", "descriptors"])


def XFeat(weights="pretrained", top_k=4096, threshold=0.05):
    model = XFeatModel(weights)

    @jax.jit
    def extract(image):
        tensor, scale = preprocess(image)
        positions, scores, descriptors = extract_core(model, tensor, top_k,
                                                      threshold)
        return positions, scores, descriptors, scale

    def call(image):
        return finalize(*extract(image))

    return call


def extract_core(model, tensor, top_k, threshold):
    feature_map, logits, heat = model(tensor)
    height, width = tensor.shape[1], tensor.shape[2]
    feature_map = features.l2_normalize(feature_map[0], axis=-1)
    heatmap = backend.compute_keypoint_heatmap(logits[0])
    grid, scores = backend.compute_dense_scores(heatmap, heat[0], threshold,
                                                height, width)
    scores, chosen = jax.lax.top_k(scores, top_k)
    positions = grid[chosen]
    descriptors = features.sample_features(feature_map, positions, height,
                                           width, "bicubic")
    return positions, scores, features.l2_normalize(descriptors, axis=-1)


def finalize(positions, scores, descriptors, scale):
    valid = scores > 0
    keypoints = positions[valid] * scale
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


def XFeatModel(weights="pretrained", name="xfeat"):
    image = Input((None, None, 3))
    normed = instance_normalize(to_grayscale(image))
    x_1 = build_block(normed, [(4, 3, 1), (8, 3, 2), (8, 3, 1), (24, 3, 2)], "block1")  # fmt: skip
    x_2 = build_block(x_1 + build_skip(normed), [(24, 3, 1), (24, 3, 1)], "block2")  # fmt: skip
    x_3 = build_block(x_2, [(64, 3, 2), (64, 3, 1), (64, 1, 1)], "block3")
    x_4 = build_block(x_3, [(64, 3, 2), (64, 3, 1), (64, 3, 1)], "block4")
    x_5 = build_block(x_4, [(128, 3, 2), (128, 3, 1), (128, 3, 1), (64, 1, 1)], "block5")  # fmt: skip
    x_4 = UpSampling2D(2, interpolation="bilinear")(x_4)
    x_5 = UpSampling2D(4, interpolation="bilinear")(x_5)
    feature_map = build_fusion(x_3 + x_4 + x_5)
    heatmap = build_heatmap_head(feature_map)
    keypoints = build_keypoint_head(normed)
    model = Model(image, [feature_map, keypoints, heatmap], name=name)
    load_weights(model, weights, "paz/models/xfeat")
    return model


def load_weights(model, weights, cache):
    if weights == "pretrained":
        asset = WEIGHTS_URL.rsplit("/", 1)[-1]
        weights = get_file(asset, WEIGHTS_URL, cache_subdir=cache)
    if weights is not None:
        model.load_weights(weights)


def to_grayscale(image):
    return ops.mean(image, axis=-1, keepdims=True)


def instance_normalize(x):
    mean = ops.mean(x, axis=(1, 2), keepdims=True)
    variance = ops.var(x, axis=(1, 2), keepdims=True)
    return (x - mean) / ops.sqrt(variance + INSTANCE_NORM_EPSILON)


def build_block(x, specs, prefix):
    for index, (filters, kernel, stride) in enumerate(specs):
        x = xfeat_layer(x, filters, kernel, stride, f"{prefix}_{index}")
    return x


def xfeat_layer(x, filters, kernel, stride, name):
    x = xfeat_conv(x, filters, kernel, stride, False, f"{name}_conv")
    x = BatchNormalization(center=False, scale=False,
                           epsilon=1e-5, name=f"{name}_bn")(x)
    return ReLU()(x)


def xfeat_conv(x, filters, kernel, stride, use_bias, name):
    x = ZeroPadding2D(1)(x) if kernel == 3 else x
    return Conv2D(filters, kernel, strides=stride, padding="valid",
                  use_bias=use_bias, name=name)(x)


def build_skip(x):
    x = AveragePooling2D(4, strides=4)(x)
    return Conv2D(24, 1, use_bias=True, name="skip_conv")(x)


def build_fusion(x):
    x = xfeat_layer(x, 64, 3, 1, "fusion_0")
    x = xfeat_layer(x, 64, 3, 1, "fusion_1")
    return Conv2D(64, 1, use_bias=True, name="fusion_out")(x)


def build_heatmap_head(x):
    x = xfeat_layer(x, 64, 1, 1, "heatmap_0")
    x = xfeat_layer(x, 64, 1, 1, "heatmap_1")
    return ops.sigmoid(Conv2D(1, 1, use_bias=True, name="heatmap_out")(x))


def build_keypoint_head(normed):
    x = ops.image.extract_patches(normed, size=8)
    x = xfeat_layer(x, 64, 1, 1, "keypoint_0")
    x = xfeat_layer(x, 64, 1, 1, "keypoint_1")
    x = xfeat_layer(x, 64, 1, 1, "keypoint_2")
    return Conv2D(65, 1, use_bias=True, name="keypoint_out")(x)
