from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jp
from keras import Model
from keras import ops
from keras.layers import Layer, Dense, LayerNormalization
from keras.utils import get_file

INPUT_DIM = 64
DESCRIPTOR_DIM = 96
NUM_LAYERS = 6
NUM_HEADS = 1
WEIGHTS_URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.25/xfeat_lighterglue_paz_jax.weights.h5"  # fmt: skip

Matches = namedtuple("Matches", ["matches0", "matches1", "scores0", "scores1"])


def LighterGlue(weights="pretrained", filter_threshold=0.1):
    model = LighterGlueModel(weights)
    forward = jax.jit(lambda inputs: model(inputs))
    prune = jax.jit(filter_matches, static_argnums=(1,))

    def call(keypoints0, descriptors0, keypoints1, descriptors1, size0, size1):
        inputs = [normalize(keypoints0, size0)[None], descriptors0[None],
                  normalize(keypoints1, size1)[None], descriptors1[None]]
        scores = forward(inputs)[0]
        return Matches(*(np.asarray(x) for x in prune(scores,
                                                      filter_threshold)))

    return call


def LighterGlueModel(weights="pretrained", name="lighterglue"):
    model = LightGlueTransformer(name=name)
    build_model(model)
    if weights == "pretrained":
        asset = WEIGHTS_URL.rsplit("/", 1)[-1]
        weights = get_file(asset, WEIGHTS_URL,
                           cache_subdir="paz/models/lightglue")
    if weights is not None:
        model.load_weights(weights)
    return model


def build_model(model):
    keypoints = ops.zeros((1, 8, 2))
    descriptors = ops.zeros((1, 8, INPUT_DIM))
    model([keypoints, descriptors, keypoints, descriptors])


class LightGlueTransformer(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_proj = Dense(DESCRIPTOR_DIM, name="input_proj")
        self.encoding = FourierEncoding(DESCRIPTOR_DIM // NUM_HEADS)
        self.self_blocks = [SelfBlock(i) for i in range(NUM_LAYERS)]
        self.cross_blocks = [CrossBlock(i) for i in range(NUM_LAYERS)]
        self.assignment = MatchAssignment()

    def call(self, inputs):
        keypoints0, descriptors0, keypoints1, descriptors1 = inputs
        cos0, sin0 = self.encoding(keypoints0)
        cos1, sin1 = self.encoding(keypoints1)
        x0 = self.input_proj(descriptors0)
        x1 = self.input_proj(descriptors1)
        for self_block, cross_block in zip(self.self_blocks, self.cross_blocks):
            x0 = self_block(x0, cos0, sin0)
            x1 = self_block(x1, cos1, sin1)
            x0, x1 = cross_block(x0, x1)
        return self.assignment(x0, x1)


class FourierEncoding(Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.projection = Dense(dim // 2, use_bias=False, name="projection")

    def call(self, keypoints):
        angles = ops.repeat(self.projection(keypoints), 2, axis=-1)
        return ops.cos(angles), ops.sin(angles)


class SelfBlock(Layer):
    def __init__(self, index, **kwargs):
        super().__init__(name=f"self_block_{index}", **kwargs)
        self.qkv = Dense(3 * DESCRIPTOR_DIM, name="qkv")
        self.out_proj = Dense(DESCRIPTOR_DIM, name="out_proj")
        self.ffn = FeedForward()

    def call(self, x, cos, sin):
        heads, groups = NUM_HEADS, 3
        qkv = split_heads(self.qkv(x), heads, groups)
        query = rotate(qkv[..., 0], cos, sin)
        key = rotate(qkv[..., 1], cos, sin)
        context = attention(query, key, qkv[..., 2])
        message = self.out_proj(merge_heads(context))
        return x + self.ffn(ops.concatenate([x, message], axis=-1))


class CrossBlock(Layer):
    def __init__(self, index, **kwargs):
        super().__init__(name=f"cross_block_{index}", **kwargs)
        self.to_qk = Dense(DESCRIPTOR_DIM, name="to_qk")
        self.to_v = Dense(DESCRIPTOR_DIM, name="to_v")
        self.out_proj = Dense(DESCRIPTOR_DIM, name="out_proj")
        self.ffn = FeedForward()

    def call(self, x0, x1):
        query0 = scale_query(split_heads(self.to_qk(x0), NUM_HEADS))
        query1 = scale_query(split_heads(self.to_qk(x1), NUM_HEADS))
        value0 = split_heads(self.to_v(x0), NUM_HEADS)
        value1 = split_heads(self.to_v(x1), NUM_HEADS)
        similarity = ops.einsum("bhid,bhjd->bhij", query0, query1)
        message0 = ops.softmax(similarity, axis=-1) @ value1
        message1 = ops.softmax(swap(similarity), axis=-1) @ value0
        return self.update(x0, message0), self.update(x1, message1)

    def update(self, x, message):
        message = self.out_proj(merge_heads(message))
        return x + self.ffn(ops.concatenate([x, message], axis=-1))


class FeedForward(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.expand = Dense(2 * DESCRIPTOR_DIM, name="expand")
        self.norm = LayerNormalization(epsilon=1e-5, name="norm")
        self.project = Dense(DESCRIPTOR_DIM, name="project")

    def call(self, x):
        x = ops.gelu(self.norm(self.expand(x)), approximate=False)
        return self.project(x)


class MatchAssignment(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_proj = Dense(DESCRIPTOR_DIM, name="final_proj")
        self.matchability = Dense(1, name="matchability")

    def call(self, x0, x1):
        scale = DESCRIPTOR_DIM ** 0.25
        matches0 = self.final_proj(x0) / scale
        matches1 = self.final_proj(x1) / scale
        similarity = ops.einsum("bmd,bnd->bmn", matches0, matches1)
        return double_softmax(similarity, self.matchability(x0),
                              self.matchability(x1))


def attention(query, key, value):
    scale = query.shape[-1] ** -0.5
    similarity = ops.einsum("bhid,bhjd->bhij", query, key) * scale
    return ops.softmax(similarity, axis=-1) @ value


def scale_query(query):
    return query * query.shape[-1] ** -0.25


def split_heads(x, num_heads, groups=1):
    batch, length = x.shape[0], x.shape[1]
    shape = (batch, length, num_heads, -1, groups)
    x = ops.reshape(x, shape if groups > 1 else shape[:-1])
    return ops.transpose(x, (0, 2, 1, 3, 4) if groups > 1 else (0, 2, 1, 3))


def merge_heads(x):
    batch, length = x.shape[0], x.shape[2]
    return ops.reshape(ops.transpose(x, (0, 2, 1, 3)), (batch, length, -1))


def rotate(x, cos, sin):
    cos, sin = ops.expand_dims(cos, 1), ops.expand_dims(sin, 1)
    return x * cos + rotate_half(x) * sin


def rotate_half(x):
    first, second = x[..., ::2], x[..., 1::2]
    return ops.reshape(ops.stack([-second, first], axis=-1), x.shape)


def swap(x):
    return ops.transpose(x, (0, 1, 3, 2))


def double_softmax(similarity, z0, z1):
    certainty = log_sigmoid(z0) + swap_last(log_sigmoid(z1))
    scores = ops.log_softmax(similarity, axis=2)
    scores = scores + ops.log_softmax(similarity, axis=1) + certainty
    top = ops.concatenate([scores, log_sigmoid(-z0)], axis=2)
    corner = ops.zeros((z0.shape[0], 1, 1))
    bottom = ops.concatenate([swap_last(log_sigmoid(-z1)), corner], axis=2)
    return ops.concatenate([top, bottom], axis=1)


def swap_last(x):
    return ops.transpose(x, (0, 2, 1))


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
