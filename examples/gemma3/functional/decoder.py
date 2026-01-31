import keras
from keras import ops
from keras.layers import Dropout
from keras.layers import EinsumDense
from keras.layers import RMSNormalization

from examples.gemma3.functional.attention import build_attention
from examples.gemma3.functional.attention import _compute_attention_mask
from examples.gemma3.functional.core import add_residual
from examples.gemma3.functional.core import clip_float16


def build_decoder_block(
    hidden_dim,
    intermediate_dim,
    head_dim,
    num_query_heads,
    num_key_value_heads,
    query_head_dim_normalize=True,
    use_query_key_norm=False,
    use_post_ffw_norm=False,
    use_post_attention_norm=False,
    gate_dim_reduction=1,
    logit_soft_cap=None,
    use_sliding_window_attention=False,
    sliding_window_size=4096,
    layer_norm_epsilon=1e-6,
    rope_wavelength=10000.0,
    rope_scaling_factor=1.0,
    use_bidirectional_attention=False,
    dropout=0.0,
    dtype=None,
    name_prefix="decoder",
):
    pre_name = name_prefix + "_pre_attention_norm"
    pre_norm = RMSNormalization(
        epsilon=layer_norm_epsilon, dtype=dtype, name=pre_name
    )
    post_norm = None
    if use_post_attention_norm:
        post_name = name_prefix + "_post_attention_norm"
        post_norm = RMSNormalization(
            epsilon=layer_norm_epsilon, dtype=dtype, name=post_name
        )

    att_name = name_prefix + "_attention"
    attention_apply, attention_layers = build_attention(
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
        att_name,
    )

    attention_dropout = None
    if dropout:
        drop_name = name_prefix + "_attention_dropout"
        attention_dropout = Dropout(rate=dropout, dtype=dtype, name=drop_name)

    ffw_name = name_prefix + "_pre_ffw_norm"
    pre_ffw_norm = RMSNormalization(
        epsilon=layer_norm_epsilon, dtype=dtype, name=ffw_name
    )
    post_ffw_norm = None
    if use_post_ffw_norm:
        post_name = name_prefix + "_post_ffw_norm"
        post_ffw_norm = RMSNormalization(
            epsilon=layer_norm_epsilon, dtype=dtype, name=post_name
        )

    feedforward_dim = intermediate_dim // gate_dim_reduction
    gate_name = name_prefix + "_ffw_gating"
    gate_name_2 = name_prefix + "_ffw_gating_2"
    linear_name = name_prefix + "_ffw_linear"
    gating_ffw = EinsumDense(
        "btd,df->btf", (None, feedforward_dim), dtype=dtype, name=gate_name
    )
    gating_ffw_2 = EinsumDense(
        "btd,df->btf", (None, feedforward_dim), dtype=dtype, name=gate_name_2
    )
    ffw_linear = EinsumDense(
        "btf,fd->btd", (None, hidden_dim), dtype=dtype, name=linear_name
    )

    layers = (pre_norm, post_norm, attention_layers, attention_dropout)
    layers = layers + (pre_ffw_norm, post_ffw_norm, gating_ffw)
    layers = layers + (gating_ffw_2, ffw_linear)

    def unpack_cache(cache):
        if cache is None:
            return None, 0, None
        return cache[0], cache[1], cache[2]

    def apply_attention_path(inputs, padding, vision, cache_state, train):
        cache_data, cache_index, cache_mask = cache_state
        hidden = pre_norm(inputs)
        mask = _compute_attention_mask(
            hidden, padding, vision, cache_data, cache_index, use_bidirectional_attention
        )
        if cache_data is None:
            out = attention_apply(hidden, mask, None, train)
            new_cache = None
        else:
            cache_arg = (cache_data, cache_index, cache_mask)
            out, new_cache = attention_apply(hidden, mask, cache_arg, train)
        if use_post_attention_norm and post_norm is not None:
            out = post_norm(out)
        if attention_dropout is not None:
            out = attention_dropout(out, training=train)
        return add_residual(inputs, out), new_cache

    def apply_ffw_path(attention_inputs):
        normalized = pre_ffw_norm(attention_inputs)
        first_gate = gating_ffw(normalized)
        second_gate = gating_ffw_2(normalized)
        hidden = keras.activations.gelu(first_gate, approximate=True)
        hidden = hidden * second_gate
        hidden = ffw_linear(hidden)
        if use_post_ffw_norm and post_ffw_norm is not None:
            hidden = post_ffw_norm(hidden)
        return add_residual(attention_inputs, hidden)

    def apply_block(inputs, padding, vision, cache, train):
        dtype_name = keras.backend.standardize_dtype(inputs.dtype)
        if dtype_name == "float16":
            inputs = clip_float16(inputs)
        cache_state = unpack_cache(cache)
        attention_inputs, new_cache = apply_attention_path(
            inputs, padding, vision, cache_state, train
        )
        output = apply_ffw_path(attention_inputs)
        if cache_state[0] is not None:
            return output, new_cache
        return output

    return apply_block, layers
