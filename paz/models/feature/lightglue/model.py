from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jp
from keras import Input, Model, ops
from keras.layers import Dense, LayerNormalization
from keras.utils import get_file

from paz.layers import SplitDim, MergeDims

INPUT_DIM = 64
DESCRIPTOR_DIM = 96
HEAD_DIM = 96
NUM_LAYERS = 6
WEIGHTS_URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.25/xfeat_lighterglue_paz_jax.weights.h5"  # fmt: skip

Matches = namedtuple("Matches", ["matches0", "matches1", "scores0", "scores1"])


def LighterGlue(weights="pretrained", filter_threshold=0.1):
    model = LighterGlueModel(weights)
    forward = jax.jit(lambda inputs: model(inputs))
    prune = jax.jit(filter_matches, static_argnums=(1,))

    def call(keypoints0, descriptors0, keypoints1, descriptors1, size0, size1):
        first = prepare(keypoints0, descriptors0, size0)
        second = prepare(keypoints1, descriptors1, size1)
        scores = forward(first + second)[0]
        outputs = prune(scores, filter_threshold)
        return Matches(*(np.asarray(value) for value in outputs))

    return call


def prepare(keypoints, descriptors, size):
    return [normalize(keypoints, size)[None], descriptors[None]]


def LighterGlueModel(weights="pretrained", name="lighterglue"):
    model = build_lighterglue(name)
    if weights == "pretrained":
        asset = WEIGHTS_URL.rsplit("/", 1)[-1]
        weights = get_file(asset, WEIGHTS_URL,
                           cache_subdir="paz/models/lightglue")
    if weights is not None:
        model.load_weights(weights)
    return model


def build_lighterglue(name):
    keypoints0, descriptors0 = Input((None, 2)), Input((None, INPUT_DIM))
    keypoints1, descriptors1 = Input((None, 2)), Input((None, INPUT_DIM))
    encode = build_encoder("encoding")
    project = Dense(DESCRIPTOR_DIM, name="input_projection")
    cosine0, sine0 = encode(keypoints0)
    cosine1, sine1 = encode(keypoints1)
    x0, x1 = project(descriptors0), project(descriptors1)
    for index in range(NUM_LAYERS):
        x0, x1 = transformer_layer(x0, x1, cosine0, sine0, cosine1, sine1, index)  # fmt: skip
    scores = assign_matches(x0, x1, "assignment")
    inputs = [keypoints0, descriptors0, keypoints1, descriptors1]
    return Model(inputs, scores, name=name)


def build_encoder(name):
    project = Dense(HEAD_DIM // 2, use_bias=False, name=f"{name}_projection")

    def encode(keypoints):
        angles = ops.repeat(project(keypoints), 2, axis=-1)
        return ops.cos(angles), ops.sin(angles)

    return encode


def transformer_layer(x0, x1, cosine0, sine0, cosine1, sine1, index):
    self_attend = build_self_attention(f"self_attention_{index}")
    x0 = self_attend(x0, cosine0, sine0)
    x1 = self_attend(x1, cosine1, sine1)
    return build_cross_attention(f"cross_attention_{index}")(x0, x1)


def build_self_attention(name):
    to_qkv = Dense(3 * DESCRIPTOR_DIM, name=f"{name}_qkv")
    project = Dense(DESCRIPTOR_DIM, name=f"{name}_projection")
    feed_forward = build_feed_forward(name)

    def attend(x, cosine, sine):
        heads = SplitDim(-1, (HEAD_DIM, 3))(to_qkv(x))
        query = rotate(heads[..., 0], cosine, sine)
        key = rotate(heads[..., 1], cosine, sine)
        message = project(scaled_attention(query, key, heads[..., 2]))
        return x + feed_forward(ops.concatenate([x, message], axis=-1))

    return attend


def build_cross_attention(name):
    to_query = Dense(DESCRIPTOR_DIM, name=f"{name}_query")
    to_value = Dense(DESCRIPTOR_DIM, name=f"{name}_value")
    project = Dense(DESCRIPTOR_DIM, name=f"{name}_projection")
    feed_forward = build_feed_forward(name)

    def attend(x0, x1):
        query0, query1 = scale(to_query(x0)), scale(to_query(x1))
        value0, value1 = to_value(x0), to_value(x1)
        similarity = ops.einsum("bid,bjd->bij", query0, query1)
        message0 = project(ops.matmul(ops.softmax(similarity, -1), value1))
        message1 = project(ops.matmul(ops.softmax(swap(similarity), -1), value0))  # fmt: skip
        x0 = x0 + feed_forward(ops.concatenate([x0, message0], axis=-1))
        x1 = x1 + feed_forward(ops.concatenate([x1, message1], axis=-1))
        return x0, x1

    return attend


def build_feed_forward(name):
    expand = Dense(2 * DESCRIPTOR_DIM, name=f"{name}_expand")
    normalize_features = LayerNormalization(epsilon=1e-5, name=f"{name}_norm")
    project = Dense(DESCRIPTOR_DIM, name=f"{name}_project")

    def transform(x):
        return project(ops.gelu(normalize_features(expand(x)), approximate=False))  # fmt: skip

    return transform


def assign_matches(x0, x1, name):
    project = Dense(DESCRIPTOR_DIM, name=f"{name}_projection")
    matchability = Dense(1, name=f"{name}_matchability")
    scale = DESCRIPTOR_DIM ** 0.25
    similarity = ops.einsum("bmd,bnd->bmn", project(x0) / scale, project(x1) / scale)  # fmt: skip
    return double_softmax(similarity, matchability(x0), matchability(x1))


def scaled_attention(query, key, value):
    scores = ops.einsum("bid,bjd->bij", query, key) / HEAD_DIM ** 0.5
    return ops.matmul(ops.softmax(scores, axis=-1), value)


def rotate(x, cosine, sine):
    return x * cosine + rotate_half(x) * sine


def rotate_half(x):
    pairs = SplitDim(-1, (HEAD_DIM // 2, 2))(x)
    return MergeDims(-2)(ops.stack([-pairs[..., 1], pairs[..., 0]], axis=-1))


def scale(query):
    return query * HEAD_DIM ** -0.25


def swap(x):
    return ops.transpose(x, (0, 2, 1))


def double_softmax(similarity, z0, z1):
    certainty = log_sigmoid(z0) + swap(log_sigmoid(z1))
    scores = ops.log_softmax(similarity, 2) + ops.log_softmax(similarity, 1)
    top = ops.concatenate([scores + certainty, log_sigmoid(-z0)], axis=2)
    bottom = ops.concatenate([swap(log_sigmoid(-z1)), corner(z0)], axis=2)
    return ops.concatenate([top, bottom], axis=1)


def corner(z0):
    return ops.zeros_like(z0[:, :1, :])


def log_sigmoid(x):
    return -ops.softplus(-x)


def normalize(keypoints, size):
    size = ops.cast(size, "float32")
    return (keypoints - size / 2) / (ops.max(size) / 2)


def filter_matches(scores, threshold):
    valid = scores[:-1, :-1]
    match0 = jp.argmax(valid, axis=1)
    match1 = jp.argmax(valid, axis=0)
    mutual0 = jp.arange(match0.shape[0]) == match1[match0]
    mutual1 = jp.arange(match1.shape[0]) == match0[match1]
    strength0 = jp.where(mutual0, jp.exp(jp.max(valid, axis=1)), 0.0)
    keep0 = mutual0 & (strength0 > threshold)
    keep1 = mutual1 & keep0[match1]
    matches0 = jp.where(keep0, match0, -1)
    matches1 = jp.where(keep1, match1, -1)
    strength1 = jp.where(mutual1, strength0[match1], 0.0)
    return matches0, matches1, strength0, strength1


def match_pairs(matches):
    source = np.where(matches.matches0 > -1)[0]
    return np.stack([source, matches.matches0[source]], axis=-1)
