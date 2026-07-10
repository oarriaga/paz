from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jp

Matches = namedtuple("Matches", ["matches0", "matches1", "scores0", "scores1"])


def LighterGlue(params, num_heads=1, filter_threshold=0.1):
    core = jax.jit(match_core, static_argnums=(7, 8))

    def call(keypoints0, descriptors0, keypoints1, descriptors1,
             size0, size1):
        outputs = core(params, keypoints0, descriptors0, keypoints1,
                       descriptors1, size0, size1, num_heads, filter_threshold)
        return Matches(*(np.asarray(value) for value in outputs))

    return call


def match_core(params, keypoints0, descriptors0, keypoints1, descriptors1,
               size0, size1, num_heads, threshold):
    scores = assignment(params, keypoints0, descriptors0, keypoints1,
                        descriptors1, size0, size1, num_heads)
    return filter_matches(scores, threshold)


def assignment(params, keypoints0, descriptors0, keypoints1, descriptors1,
               size0, size1, num_heads):
    keypoints0 = normalize_keypoints(keypoints0, size0)
    keypoints1 = normalize_keypoints(keypoints1, size1)
    cos0, sin0 = position_encoding(params.posenc, keypoints0)
    cos1, sin1 = position_encoding(params.posenc, keypoints1)
    x0 = linear(params.input_proj, descriptors0)
    x1 = linear(params.input_proj, descriptors1)
    for layer in params.layers:
        x0 = self_attention(layer.self_attn, x0, cos0, sin0, num_heads)
        x1 = self_attention(layer.self_attn, x1, cos1, sin1, num_heads)
        x0, x1 = cross_attention(layer.cross_attn, x0, x1, num_heads)
    return log_assignment(params.assign, x0, x1)


def normalize_keypoints(keypoints, size):
    size = jp.asarray(size, keypoints.dtype)
    center = keypoints - size / 2
    return center / (jp.max(size) / 2)


def position_encoding(weight, keypoints):
    projected = keypoints @ weight
    angles = jp.repeat(projected, 2, axis=-1)
    return jp.cos(angles), jp.sin(angles)


def self_attention(layer, x, cos, sin, num_heads):
    qkv = split_heads(linear(layer.qkv, x), num_heads, 3)
    query, key, value = qkv[..., 0], qkv[..., 1], qkv[..., 2]
    query = apply_rotary(query, cos, sin)
    key = apply_rotary(key, cos, sin)
    message = merge_heads(attention(query, key, value))
    message = linear(layer.out_proj, message)
    return x + feed_forward(layer.ffn, jp.concatenate([x, message], -1))


def cross_attention(layer, x0, x1, num_heads):
    query0 = scale_query(split_heads(linear(layer.qk, x0), num_heads))
    query1 = scale_query(split_heads(linear(layer.qk, x1), num_heads))
    value0 = split_heads(linear(layer.v, x0), num_heads)
    value1 = split_heads(linear(layer.v, x1), num_heads)
    similarity = jp.einsum("hid,hjd->hij", query0, query1)
    message0 = jax.nn.softmax(similarity, -1) @ value1
    message1 = jax.nn.softmax(jp.swapaxes(similarity, -1, -2), -1) @ value0
    x0 = x0 + update(layer, x0, message0)
    x1 = x1 + update(layer, x1, message1)
    return x0, x1


def update(layer, x, message):
    message = linear(layer.out_proj, merge_heads(message))
    return feed_forward(layer.ffn, jp.concatenate([x, message], -1))


def scale_query(query):
    head_dim = query.shape[-1]
    return query * head_dim ** -0.25


def attention(query, key, value):
    scale = query.shape[-1] ** -0.5
    similarity = jp.einsum("hid,hjd->hij", query, key) * scale
    return jax.nn.softmax(similarity, -1) @ value


def split_heads(x, num_heads, groups=1):
    length = x.shape[0]
    x = x.reshape(length, num_heads, -1, groups) if groups > 1 else \
        x.reshape(length, num_heads, -1)
    return jp.moveaxis(x, 0, 1)


def merge_heads(x):
    length = x.shape[1]
    return jp.moveaxis(x, 0, 1).reshape(length, -1)


def apply_rotary(x, cos, sin):
    return x * cos + rotate_half(x) * sin


def rotate_half(x):
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    rotated = jp.stack([-pairs[..., 1], pairs[..., 0]], axis=-1)
    return rotated.reshape(x.shape)


def feed_forward(params, x):
    x = linear(params.input, x)
    x = layer_norm(params.norm, x)
    return linear(params.output, jax.nn.gelu(x, approximate=False))


def layer_norm(params, x):
    mean = jp.mean(x, axis=-1, keepdims=True)
    variance = jp.var(x, axis=-1, keepdims=True)
    normed = (x - mean) / jp.sqrt(variance + 1e-5)
    return normed * params.weight + params.bias


def linear(params, x):
    return x @ params.weight + params.bias


def log_assignment(params, x0, x1):
    scale = x0.shape[-1] ** 0.25
    matches0 = linear(params.final_proj, x0) / scale
    matches1 = linear(params.final_proj, x1) / scale
    similarity = matches0 @ matches1.T
    z0 = linear(params.matchability, x0)
    z1 = linear(params.matchability, x1)
    return double_softmax(similarity, z0, z1)


def double_softmax(similarity, z0, z1):
    rows, columns = similarity.shape
    certainty = log_sigmoid(z0) + log_sigmoid(z1).T
    scores = jax.nn.log_softmax(similarity, 1) + \
        jax.nn.log_softmax(similarity, 0) + certainty
    output = jp.zeros((rows + 1, columns + 1))
    output = output.at[:rows, :columns].set(scores)
    output = output.at[:rows, columns].set(log_sigmoid(-z0)[:, 0])
    return output.at[rows, :columns].set(log_sigmoid(-z1)[:, 0])


def log_sigmoid(x):
    return -jax.nn.softplus(-x)


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
