from collections import namedtuple
from functools import partial

import numpy as np
import jax
import jax.numpy as jp
from keras import Input, Model, ops
from keras.layers import Dense, LayerNormalization
from keras.utils import get_file

from paz.layers import SplitDim, MergeDims

WEIGHTS_URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.25/xfeat_lighterglue_paz_jax.weights.h5"  # fmt: skip

Matches = namedtuple(
    "Matches", ["matches_0", "matches_1", "scores_0", "scores_1"])


def LighterGlue(weights="pretrained", filter_threshold=0.1, capacity=4096):
    model = LighterGlueModel(weights)

    @jax.jit
    def match(keypoints_0, descriptors_0, mask_0, keypoints_1,
              descriptors_1, mask_1, size_0, size_1):
        first = prepare(keypoints_0, descriptors_0, mask_0, size_0)
        second = prepare(keypoints_1, descriptors_1, mask_1, size_1)
        scores = model(first + second)[0]
        return filter_matches(scores, mask_0, mask_1, filter_threshold)

    def call(keypoints_0, descriptors_0, keypoints_1, descriptors_1,
             size_0, size_1):
        count_0 = min(len(keypoints_0), capacity)
        count_1 = min(len(keypoints_1), capacity)
        first = pad_features(keypoints_0, descriptors_0, capacity)
        second = pad_features(keypoints_1, descriptors_1, capacity)
        outputs = match(*first, *second, size_0, size_1)
        matches_0, matches_1, scores_0, scores_1 = outputs
        matches_0, scores_0 = matches_0[:count_0], scores_0[:count_0]
        matches_1, scores_1 = matches_1[:count_1], scores_1[:count_1]
        return Matches(matches_0, matches_1, scores_0, scores_1)

    return call


def prepare(keypoints, descriptors, mask, size):
    return [normalize(keypoints, size)[None], descriptors[None], mask[None]]


def pad_features(keypoints, descriptors, capacity):
    count = min(keypoints.shape[0], capacity)
    keypoints = jp.asarray(keypoints)[:count]
    descriptors = jp.asarray(descriptors)[:count]
    mask = jp.zeros(capacity).at[:count].set(1.0)
    keypoints = jp.zeros((capacity, 2)).at[:count].set(keypoints)
    descriptors = jp.zeros((capacity, descriptors.shape[1])).at[:count].set(descriptors)  # fmt: skip
    return keypoints, descriptors, mask


def LighterGlueModel(weights="pretrained", name="lighterglue"):
    input_dim, dim, num_layers = 64, 96, 6
    keypoints_0, descriptors_0 = Input((None, 2)), Input((None, input_dim))
    keypoints_1, descriptors_1 = Input((None, 2)), Input((None, input_dim))
    mask_0, mask_1 = Input((None,)), Input((None,))
    encode = build_encoder(dim, "encoding")
    project = Dense(dim, name="input_projection")
    cos_0, sin_0 = encode(keypoints_0)
    cos_1, sin_1 = encode(keypoints_1)
    x_0, x_1 = project(descriptors_0), project(descriptors_1)
    arguments = (cos_0, sin_0, cos_1, sin_1, mask_0, mask_1, dim)
    transform = partial(transformer_layer, *arguments)
    for arg in range(num_layers):
        x_0, x_1 = transform(arg, x_0, x_1)
    scores = assign_matches(x_0, x_1, mask_0, mask_1, dim, "assignment")
    inputs = [keypoints_0, descriptors_0, mask_0,
              keypoints_1, descriptors_1, mask_1]
    model = Model(inputs, scores, name=name)
    load_weights(model, weights, "paz/models/lightglue")
    return model


def load_weights(model, weights, cache):
    if weights == "pretrained":
        asset = WEIGHTS_URL.rsplit("/", 1)[-1]
        weights = get_file(asset, WEIGHTS_URL, cache_subdir=cache)
    if weights is not None:
        model.load_weights(weights)


def build_encoder(dim, name):
    project = Dense(dim // 2, use_bias=False, name=f"{name}_projection")

    def encode(keypoints):
        angles = ops.repeat(project(keypoints), 2, axis=-1)
        return ops.cos(angles), ops.sin(angles)

    return encode


def transformer_layer(cos_0, sin_0, cos_1, sin_1, mask_0, mask_1, dim, arg,
                      x_0, x_1):
    self_attend = build_self_attention(dim, f"self_attention_{arg}")
    x_0 = self_attend(x_0, cos_0, sin_0, mask_0)
    x_1 = self_attend(x_1, cos_1, sin_1, mask_1)
    cross_attend = build_cross_attention(dim, f"cross_attention_{arg}")
    return cross_attend(x_0, x_1, mask_0, mask_1)


def build_self_attention(dim, name):
    to_qkv = Dense(3 * dim, name=f"{name}_qkv")
    project = Dense(dim, name=f"{name}_projection")
    feed_forward = build_feed_forward(dim, name)

    def attend(x, cos, sin, mask):
        heads = SplitDim(-1, (dim, 3))(to_qkv(x))
        query = rotate(heads[..., 0], cos, sin)
        key = rotate(heads[..., 1], cos, sin)
        message = project(scaled_attention(query, key, heads[..., 2], mask))
        return x + feed_forward(ops.concatenate([x, message], axis=-1))

    return attend


def build_cross_attention(dim, name):
    to_query = Dense(dim, name=f"{name}_query")
    to_value = Dense(dim, name=f"{name}_value")
    project = Dense(dim, name=f"{name}_projection")
    feed_forward = build_feed_forward(dim, name)

    def attend(x_0, x_1, mask_0, mask_1):
        query_0, query_1 = scale(to_query(x_0)), scale(to_query(x_1))
        value_0, value_1 = to_value(x_0), to_value(x_1)
        similarity = ops.einsum("bid,bjd->bij", query_0, query_1)
        attention_0 = ops.softmax(similarity + key_bias(mask_1), -1)
        attention_1 = ops.softmax(swap(similarity) + key_bias(mask_0), -1)
        message_0 = project(ops.matmul(attention_0, value_1))
        message_1 = project(ops.matmul(attention_1, value_0))
        x_0 = x_0 + feed_forward(ops.concatenate([x_0, message_0], axis=-1))
        x_1 = x_1 + feed_forward(ops.concatenate([x_1, message_1], axis=-1))
        return x_0, x_1

    return attend


def build_feed_forward(dim, name):
    expand = Dense(2 * dim, name=f"{name}_expand")
    normalize_features = LayerNormalization(epsilon=1e-5, name=f"{name}_norm")
    project = Dense(dim, name=f"{name}_project")

    def transform(x):
        return project(ops.gelu(normalize_features(expand(x)), approximate=False))  # fmt: skip

    return transform


def assign_matches(x_0, x_1, mask_0, mask_1, dim, name):
    project = Dense(dim, name=f"{name}_projection")
    matchability = Dense(1, name=f"{name}_matchability")
    scale = dim ** 0.25
    similarity = ops.einsum("bmd,bnd->bmn", project(x_0) / scale, project(x_1) / scale)  # fmt: skip
    similarity = similarity + pair_bias(mask_0, mask_1)
    return double_softmax(similarity, matchability(x_0), matchability(x_1))


def scaled_attention(query, key, value, mask):
    scores = ops.einsum("bid,bjd->bij", query, key) / query.shape[-1] ** 0.5
    return ops.matmul(ops.softmax(scores + key_bias(mask), axis=-1), value)


def key_bias(mask):
    return ops.where(mask[:, None, :] > 0, 0.0, -1e9)


def pair_bias(mask_0, mask_1):
    valid = (mask_0[:, :, None] > 0) & (mask_1[:, None, :] > 0)
    return ops.where(valid, 0.0, -1e9)


def rotate(x, cos, sin):
    return x * cos + rotate_half(x) * sin


def rotate_half(x):
    pairs = SplitDim(-1, (x.shape[-1] // 2, 2))(x)
    return MergeDims(-2)(ops.stack([-pairs[..., 1], pairs[..., 0]], axis=-1))


def scale(query):
    return query * query.shape[-1] ** -0.25


def swap(x):
    return ops.transpose(x, (0, 2, 1))


def double_softmax(similarity, z_0, z_1):
    certainty = log_sigmoid(z_0) + swap(log_sigmoid(z_1))
    scores = ops.log_softmax(similarity, 2) + ops.log_softmax(similarity, 1)
    top = ops.concatenate([scores + certainty, log_sigmoid(-z_0)], axis=2)
    bottom = ops.concatenate([swap(log_sigmoid(-z_1)), corner(z_0)], axis=2)
    return ops.concatenate([top, bottom], axis=1)


def corner(z_0):
    return ops.zeros_like(z_0[:, :1, :])


def log_sigmoid(x):
    return -ops.softplus(-x)


def normalize(keypoints, size):
    size = ops.cast(size, "float32")
    return (keypoints - size / 2) / (ops.max(size) / 2)


def filter_matches(scores, mask_0, mask_1, threshold):
    valid = scores[:-1, :-1]
    match_0 = jp.argmax(valid, axis=1)
    match_1 = jp.argmax(valid, axis=0)
    mutual_0 = (jp.arange(match_0.shape[0]) == match_1[match_0]) & (mask_0 > 0)
    mutual_1 = (jp.arange(match_1.shape[0]) == match_0[match_1]) & (mask_1 > 0)
    strength_0 = jp.where(mutual_0, jp.exp(jp.max(valid, axis=1)), 0.0)
    keep_0 = mutual_0 & (strength_0 > threshold)
    keep_1 = mutual_1 & keep_0[match_1]
    matches_0 = jp.where(keep_0, match_0, -1)
    matches_1 = jp.where(keep_1, match_1, -1)
    strength_1 = jp.where(mutual_1, strength_0[match_1], 0.0)
    return matches_0, matches_1, strength_0, strength_1


def match_pairs(matches):
    source = np.where(matches.matches_0 > -1)[0]
    return np.stack([source, matches.matches_0[source]], axis=-1)
