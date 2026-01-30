import numpy as np
from keras import ops

from examples.gemma3.functional.gemma3 import build_gemma3_attention
from examples.gemma3.functional.test_utils import copy_attention_weights

from keras_hub.src.models.gemma3.gemma3_attention import CachedGemma3Attention


def _build_attention():
    hidden_dim = 16
    head_dim = 4
    num_query_heads = 4
    num_key_value_heads = 2
    use_query_key_norm = True
    query_head_dim_normalize = True
    use_sliding_window_attention = False
    sliding_window_size = 8
    rope_wavelength = 10000.0
    rope_scaling_factor = 1.0
    logit_soft_cap = None
    use_bidirectional_attention = False
    layer_norm_epsilon = 1e-6
    dropout = 0.0
    dtype = "float32"
    name_prefix = "test"
    return build_gemma3_attention(
        hidden_dim,
        head_dim,
        num_query_heads,
        num_key_value_heads,
        use_query_key_norm,
        query_head_dim_normalize,
        use_sliding_window_attention,
        sliding_window_size,
        rope_wavelength,
        rope_scaling_factor,
        logit_soft_cap,
        use_bidirectional_attention,
        layer_norm_epsilon,
        dropout,
        dtype,
        name_prefix,
    )


def _build_hub_attention(name):
    return CachedGemma3Attention(
        head_dim=4,
        num_query_heads=4,
        num_key_value_heads=2,
        use_query_key_norm=True,
        logit_soft_cap=None,
        use_sliding_window_attention=False,
        sliding_window_size=8,
        query_head_dim_normalize=True,
        rope_wavelength=10000.0,
        rope_scaling_factor=1.0,
        dropout=0.0,
        name=name,
    )


def test_attention_output_matches_hub():
    rng = np.random.default_rng(5)
    shape = (2, 3, 16)
    values = rng.standard_normal(shape).astype("float32")
    inputs = ops.convert_to_tensor(values)
    attention_mask = ops.ones((2, 3, 3), dtype="bool")

    hub_layer = _build_hub_attention("hub")
    hub_layer(inputs, attention_mask=attention_mask)

    apply_attention, layers = _build_attention()
    apply_attention(inputs, attention_mask, None, False)

    copy_attention_weights(layers, hub_layer)

    hub_output = hub_layer(inputs, attention_mask=attention_mask)
    clean_output = apply_attention(inputs, attention_mask, None, False)
    clean_np = ops.convert_to_numpy(clean_output)
    hub_np = ops.convert_to_numpy(hub_output)
    np.testing.assert_allclose(clean_np, hub_np, rtol=1e-5, atol=1e-5)


def test_attention_cache_matches_hub():
    rng = np.random.default_rng(7)
    shape = (2, 1, 16)
    values = rng.standard_normal(shape).astype("float32")
    inputs = ops.convert_to_tensor(values)
    attention_mask = ops.ones((2, 1, 4), dtype="bool")
    cache = ops.zeros((2, 2, 4, 2, 4), dtype="float32")
    cache_update_mask = ops.ones((2, 1), dtype="bool")

    hub_layer = _build_hub_attention("hub_cache")
    hub_layer(
        inputs,
        attention_mask=attention_mask,
        cache=cache,
        cache_update_index=2,
        cache_update_mask=cache_update_mask,
    )

    apply_attention, layers = _build_attention()
    cache_state = (cache, 2, cache_update_mask)
    apply_attention(inputs, attention_mask, cache_state, False)

    copy_attention_weights(layers, hub_layer)

    _, hub_cache = hub_layer(
        inputs,
        attention_mask=attention_mask,
        cache=cache,
        cache_update_index=2,
        cache_update_mask=cache_update_mask,
    )
    _, clean_cache = apply_attention(inputs, attention_mask, cache_state, False)
    clean_np = ops.convert_to_numpy(clean_cache)
    hub_np = ops.convert_to_numpy(hub_cache)
    np.testing.assert_allclose(clean_np, hub_np, rtol=1e-5, atol=1e-5)
