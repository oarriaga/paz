import numpy as np
from keras import Input, Model, ops

from paz.models.transformers.attention import attend, kv_attend, build_cache
from paz.models.transformers.attention import masked_attend
from paz.models.transformers.attention import split_query_key_value
from paz.models.transformers.attention import compute_attention
from paz.models.transformers.attention import merge_attention_heads
from paz.models.transformers.attention import normalize_query_key
from paz.models.transformers.attention import compute_masked_attention
from paz.models.transformers.attention import rms_normalize_query_key


def build_random_tokens(shape, seed):
    values = np.random.default_rng(seed).normal(size=shape)
    return values.astype("float32")


def test_attend_reorders_output_with_the_queries():
    query = Input((4, 16), name="reorder_query")
    value = Input((5, 16), name="reorder_value")
    output = attend(query, value, 2, 8, 0.0, "reorder_test")
    model = Model([query, value], output)
    queries = build_random_tokens((1, 4, 16), 0)
    values = build_random_tokens((1, 5, 16), 1)
    order = [2, 0, 3, 1]
    attended = np.array(model([queries, values]))
    reordered = np.array(model([queries[:, order], values]))
    assert not np.allclose(attended[:, 0], attended[:, 1], atol=1e-6)
    assert np.allclose(reordered, attended[:, order], atol=1e-6)


def test_masked_attend_ignores_masked_keys():
    x = Input((4, 16), name="masked_keys_x")
    mask = Input((4,), name="masked_keys_mask")
    y = masked_attend(x, x, mask, 2, 8, 0.0, "masked_keys_test")
    model = Model([x, mask], y)
    values = np.random.default_rng(0).normal(size=(1, 4, 16))
    values = values.astype("float32")
    altered = np.copy(values)
    altered[0, 3] = 100.0
    mask_values = np.array([[1.0, 1.0, 1.0, 0.0]], "float32")
    kept = model.predict([values, mask_values], verbose=0)
    changed = model.predict([altered, mask_values], verbose=0)
    assert np.allclose(kept[:, :3], changed[:, :3], atol=1e-6)


def copy_weights_by_layer_name(source, target):
    for layer in source.layers:
        if layer.weights:
            target.get_layer(layer.name).set_weights(layer.get_weights())


def test_kv_attend_over_built_cache_matches_attend():
    query = Input((3, 16), name="cross_query")
    value = Input((5, 16), name="cross_value")
    direct = attend(query, value, 2, 8, 0.0, "cross_test")
    direct_model = Model([query, value], direct)
    cache = build_cache(value, None, 2, 8, "cross_test")
    args = (query, cache, None, None, None, 2, 16, 0.0, "cross_test")
    cached, _ = kv_attend(*args)
    cached_model = Model([query, value], cached)
    copy_weights_by_layer_name(direct_model, cached_model)
    queries = build_random_tokens((1, 3, 16), 2)
    values = build_random_tokens((1, 5, 16), 3)
    expected = np.array(direct_model([queries, values]))
    output = np.array(cached_model([queries, values]))
    assert np.allclose(output, expected, atol=1e-6)


def test_split_recovers_fused_qkv_layout():
    num_heads, head_dim = 2, 4
    hidden = num_heads * head_dim
    ramp = np.arange(3 * hidden, dtype="float32").reshape(1, 1, 3 * hidden)
    fused = ops.array(ramp)
    query, key, value = split_query_key_value(fused, num_heads, head_dim)
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


def test_merge_attention_heads_concatenates_heads_per_token():
    num_heads, num_tokens, head_dim = 2, 3, 4
    size = num_heads * num_tokens * head_dim
    context = np.arange(size, dtype="float32")
    context = context.reshape(1, num_heads, num_tokens, head_dim)
    merged = np.array(merge_attention_heads(ops.array(context)))
    expected = np.concatenate([context[:, 0], context[:, 1]], axis=-1)
    assert np.array_equal(merged, expected)


def test_normalize_query_key_normalizes_over_head_dim():
    values = np.random.RandomState(2).randn(2, 3, 5, 16).astype("float32")
    query, key = normalize_query_key(ops.array(values), ops.array(values),
                                     1e-5, "blk")
    query = np.array(query)
    assert np.allclose(query.mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(query.std(axis=-1), 1.0, atol=1e-2)
    assert np.allclose(query, np.array(key))


def test_compute_masked_attention_ignores_masked_keys():
    rng = np.random.default_rng(1)
    query = ops.array(rng.standard_normal((1, 2, 3, 8)).astype("float32"))
    key = ops.array(rng.standard_normal((1, 2, 5, 8)).astype("float32"))
    value = ops.array(rng.standard_normal((1, 2, 5, 8)).astype("float32"))
    mask = ops.array(np.array([[[[True, True, True, False, False]]]]))
    masked = compute_masked_attention(query, key, value, mask)
    reference = compute_attention(query, key[:, :, :3], value[:, :, :3])
    assert np.allclose(np.array(masked), np.array(reference), atol=1e-6)


def test_rms_normalize_query_key_keeps_shape_and_scale():
    values = np.random.RandomState(3).randn(2, 3, 5, 16).astype("float32")
    args = (ops.array(values), ops.array(values), "rms_blk")
    query, key = rms_normalize_query_key(*args)
    root_mean_square = np.sqrt((np.array(query) ** 2).mean(axis=-1))
    assert np.allclose(root_mean_square, 1.0, atol=1e-3)
    assert np.allclose(np.array(query), np.array(key))
