import numpy as np
from keras import ops

from paz.models.transformers.attention import attend, kv_attend, build_cache
from paz.models.transformers.attention import masked_attend


def test_attend_preserves_query_shape():
    x = ops.zeros((1, 4, 16))
    output = attend(x, x, 2, 8, 0.0, "attend_test")
    assert tuple(output.shape) == (1, 4, 16)


def test_masked_attend_preserves_query_shape():
    x = ops.zeros((1, 4, 16))
    mask = ops.ones((1, 4))
    output = masked_attend(x, x, mask, 2, 8, 0.0, "masked_test")
    assert tuple(output.shape) == (1, 4, 16)


def test_masked_attend_ignores_masked_keys():
    from keras import Input, Model
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
