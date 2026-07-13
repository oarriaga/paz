import numpy as np
from keras import ops

from paz.models.transformers.attention import attend, kv_attend, build_cache
from paz.models.transformers.attention import project_query_key_value
from paz.models.transformers.attention import split_query_key_value
from paz.models.transformers.attention import compute_attention
from paz.models.transformers.attention import merge_attention_heads
from paz.models.transformers.attention import normalize_query_key


def test_attend_preserves_query_shape():
    x = ops.zeros((1, 4, 16))
    output = attend(x, x, 2, 8, 0.0, "attend_test")
    assert tuple(output.shape) == (1, 4, 16)


def test_build_cache_stacks_key_and_value():
    encoder_output = ops.zeros((1, 5, 16))
    cache = build_cache(encoder_output, None, 2, 8, "cache_test")
    assert tuple(cache.shape) == (1, 2, 5, 2, 8)


def test_kv_attend_cross_attention_shape():
    encoder_output = ops.zeros((1, 5, 16))
    cache = build_cache(encoder_output, None, 2, 8, "cross_test")
    query = ops.zeros((1, 1, 16))
    args = (query, cache, None, None, None, 2, 16, 0.0, "cross_test")
    output, updated = kv_attend(*args)
    assert tuple(output.shape) == (1, 1, 16)


def test_project_query_key_value_triples_last_axis():
    tokens = ops.zeros((1, 4, 16))
    fused = project_query_key_value(tokens, 16, True, "qkv_test")
    assert tuple(fused.shape) == (1, 4, 48)


def test_split_query_key_value_shapes():
    fused = ops.zeros((1, 4, 48))
    query, key, value = split_query_key_value(fused, 2, 8)
    assert tuple(query.shape) == (1, 2, 4, 8)
    assert tuple(key.shape) == (1, 2, 4, 8)
    assert tuple(value.shape) == (1, 2, 4, 8)


def test_split_recovers_fused_qkv_layout():
    num_heads, head_dim = 2, 4
    hidden = num_heads * head_dim
    ramp = np.arange(3 * hidden, dtype="float32").reshape(1, 1, 3 * hidden)
    query, key, value = split_query_key_value(ops.array(ramp), num_heads, head_dim)
    expected = ramp.reshape(3, num_heads, head_dim)
    assert np.allclose(np.array(query)[0, :, 0, :], expected[0])
    assert np.allclose(np.array(key)[0, :, 0, :], expected[1])
    assert np.allclose(np.array(value)[0, :, 0, :], expected[2])


def test_compute_attention_matches_numpy():
    rng = np.random.default_rng(0)
    shape = (1, 2, 4, 8)
    query = rng.standard_normal(shape).astype("float32")
    key = rng.standard_normal(shape).astype("float32")
    value = rng.standard_normal(shape).astype("float32")
    output = np.array(compute_attention(*map(ops.array, (query, key, value))))
    scale = shape[-1] ** -0.5
    scores = query @ np.swapaxes(key, -1, -2) * scale
    scores = scores - scores.max(axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    assert np.allclose(output, probabilities @ value, atol=1e-5)


def test_merge_attention_heads_inverts_split_layout():
    num_heads, head_dim = 2, 8
    context = ops.zeros((1, num_heads, 4, head_dim))
    merged = merge_attention_heads(context)
    assert tuple(merged.shape) == (1, 4, num_heads * head_dim)


def test_normalize_query_key_normalizes_over_head_dim():
    values = np.random.RandomState(2).randn(2, 3, 5, 16).astype("float32")
    query, key = normalize_query_key(ops.array(values), ops.array(values),
                                     1e-5, "blk")
    query = np.array(query)
    assert np.allclose(query.mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(query.std(axis=-1), 1.0, atol=1e-2)
    assert np.allclose(query, np.array(key))
