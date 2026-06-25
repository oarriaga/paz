from keras import ops

from paz.models.transformers.attention import attend, kv_attend, build_cache


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
