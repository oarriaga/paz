import keras
from keras.layers import Dropout, EinsumDense

from paz.models.foundation.gemma4.configuration import (
    build_cache_head_dim, build_feedforward_dim, build_head_dim,
    build_partial_rotary_factor, build_rope_scaling_factor,
    build_rope_wavelength, is_global_attention_layer, use_sliding_window)
from paz.models.transformers.numerics import add_residual, clip_float16
from paz.models.foundation.gemma4.layers.attention import attend, cached_attend
from paz.models.foundation.gemma4.layers.core import build_attention_mask
from paz.models.foundation.gemma4.layers.normalization import (
    build_rms_norm, build_scalar_multiply)


def decoder_block(x, padding_mask, config, layer_index, name,
                  per_layer_embedding=None, shared_kv=None):
    dtype = keras.backend.standardize_dtype(x.dtype)
    if dtype == "float16":
        x = clip_float16(x)
    hidden, kv = attention_path(
        x, padding_mask, config, layer_index, name, shared_kv=shared_kv)
    hidden = feedforward_path(hidden, config, name, layer_index=layer_index)
    # Per-layer input applied AFTER FFW, BEFORE layer scalar.
    if per_layer_embedding is not None:
        hidden = per_layer_input_path(
            hidden, per_layer_embedding, config, name)
    scaled = build_scalar_multiply("{}_layer_scalar".format(name))(hidden)
    return scaled, kv


def per_layer_input_path(x, per_layer_embed, config, name):
    epsilon, dtype = config.layer_norm_epsilon, config.dtype
    per_layer_dim = config.hidden_size_per_layer_input
    gate_name = "{}_per_layer_gate".format(name)
    gate = keras.activations.gelu(
        build_einsum_dense("btd,dp->btp", per_layer_dim, dtype, gate_name)(x),
        approximate=True)
    hidden = gate * per_layer_embed
    proj_name = "{}_per_layer_projection".format(name)
    hidden = build_einsum_dense(
        "btp,pd->btd", config.hidden_dim, dtype, proj_name)(hidden)
    norm_name = "{}_post_per_layer_norm".format(name)
    hidden = build_rms_norm(epsilon, dtype, norm_name)(hidden)
    return add_residual(x, hidden)


def attention_path(x, padding_mask, config, layer_index, name,
                   shared_kv=None):
    epsilon, dtype = config.layer_norm_epsilon, config.dtype
    norm_name = "{}_pre_attention_norm".format(name)
    hidden = build_rms_norm(epsilon, dtype, norm_name)(x)
    mask = build_block_attention_mask(padding_mask, config, layer_index)
    attention_args = build_attend_args(hidden, mask, config, layer_index, name)
    hidden, kv = attend(*attention_args, shared_kv=shared_kv)
    post_name = "{}_post_attention_norm".format(name)
    hidden = build_rms_norm(epsilon, dtype, post_name)(hidden)
    if config.dropout:
        drop_name = "{}_attention_dropout".format(name)
        hidden = Dropout(config.dropout, dtype=dtype, name=drop_name)(hidden)
    return add_residual(x, hidden), kv


def feedforward_path(x, config, name, layer_index=None):
    epsilon, dtype = config.layer_norm_epsilon, config.dtype
    feedforward_dim = (build_feedforward_dim(config, layer_index)
                       if layer_index is not None else config.intermediate_dim)
    pre_name = "{}_pre_ffw_norm".format(name)
    hidden = build_rms_norm(epsilon, dtype, pre_name)(x)
    up_equation = "btd,df->btf"
    gate_name = "{}_ffw_gating".format(name)
    gate = build_einsum_dense(
        up_equation, feedforward_dim, dtype, gate_name)(hidden)
    gate_2_name = "{}_ffw_gating_2".format(name)
    value = build_einsum_dense(
        up_equation, feedforward_dim, dtype, gate_2_name)(hidden)
    hidden = keras.activations.gelu(gate, approximate=True) * value
    down_equation = "btf,fd->btd"
    linear_name = "{}_ffw_linear".format(name)
    layer = build_einsum_dense(
        down_equation, config.hidden_dim, dtype, linear_name)
    hidden = layer(hidden)
    post_name = "{}_post_ffw_norm".format(name)
    hidden = build_rms_norm(epsilon, dtype, post_name)(hidden)
    if config.dropout:
        drop_name = "{}_ffw_dropout".format(name)
        hidden = Dropout(config.dropout, dtype=dtype, name=drop_name)(hidden)
    return add_residual(x, hidden)


def build_block_attention_mask(padding_mask, config, layer_index):
    is_global = is_global_attention_layer(config, layer_index)
    window = None
    if use_sliding_window(config, is_global):
        window = config.sliding_window_size
    return build_attention_mask(
        padding_mask, config.use_bidirectional_attention, window)


def build_attend_args(hidden, mask, config, layer_index, name):
    is_global = is_global_attention_layer(config, layer_index)
    attention_name = "{}_attention".format(name)
    return (
        hidden,
        mask,
        build_head_dim(config, is_global),
        config.num_query_heads,
        config.num_key_value_heads,
        config.layer_norm_epsilon,
        build_rope_wavelength(config, is_global),
        build_rope_scaling_factor(config, is_global),
        build_partial_rotary_factor(config, is_global),
        config.attention_logit_soft_cap,
        config.dropout,
        config.dtype,
        attention_name,
    )


def cached_decoder_block(x, cache, cache_index, config,
                          layer_index, name,
                          per_layer_embedding=None,
                          shared_kv_cache=None, positions=None):
    dtype = keras.backend.standardize_dtype(x.dtype)
    if dtype == "float16":
        x = clip_float16(x)
    hidden, cache = cached_attention_path(
        x, cache, cache_index, config, layer_index, name,
        shared_kv_cache=shared_kv_cache, positions=positions)
    hidden = feedforward_path(
        hidden, config, name, layer_index=layer_index)
    # Per-layer input applied AFTER FFW, BEFORE layer scalar.
    if per_layer_embedding is not None:
        hidden = per_layer_input_path(
            hidden, per_layer_embedding, config, name)
    scaled = build_scalar_multiply(
        "{}_layer_scalar".format(name))(hidden)
    return scaled, cache


def cached_attention_path(x, cache, cache_index, config, layer_index, name,
                           shared_kv_cache=None, positions=None):
    epsilon, dtype = config.layer_norm_epsilon, config.dtype
    norm_name = "{}_pre_attention_norm".format(name)
    hidden = build_rms_norm(epsilon, dtype, norm_name)(x)
    attention_args = build_cached_attend_args(
        hidden, cache, cache_index, config, layer_index, name, positions)
    hidden, cache = cached_attend(
        *attention_args, shared_kv_cache=shared_kv_cache)
    post_name = "{}_post_attention_norm".format(name)
    hidden = build_rms_norm(epsilon, dtype, post_name)(hidden)
    return add_residual(x, hidden), cache


def build_cached_attend_args(hidden, cache, index, config, layer_index, name,
                             positions=None):
    is_global = is_global_attention_layer(config, layer_index)
    window = None
    if use_sliding_window(config, is_global):
        window = config.sliding_window_size
    attention_name = "{}_attention".format(name)
    return (
        hidden,
        cache,
        index,
        build_head_dim(config, is_global),
        config.num_query_heads,
        config.num_key_value_heads,
        config.layer_norm_epsilon,
        build_rope_wavelength(config, is_global),
        build_rope_scaling_factor(config, is_global),
        build_partial_rotary_factor(config, is_global),
        config.attention_logit_soft_cap,
        config.dtype,
        attention_name,
        build_cache_head_dim(config),
        window,
        positions,
    )


def build_einsum_dense(equation, output_dim, dtype, name):
    shape = (None, output_dim)
    return EinsumDense(equation, shape, dtype=dtype, name=name)
