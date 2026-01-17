from collections import namedtuple

import keras
from keras import ops

from examples.gemma3.functional.attention import (
    apply_gemma3_attention,
    build_attention_config,
    build_attention_layers,
)
from examples.gemma3.functional.core import (
    build_rms_norm,
    clip_float16,
    compute_causal_mask,
    merge_padding_and_attention_mask,
)


DecoderBlockConfig = namedtuple(
    "DecoderBlockConfig",
    [
        "hidden_dim",
        "intermediate_dim",
        "head_dim",
        "num_query_heads",
        "num_key_value_heads",
        "query_head_dim_normalize",
        "use_query_key_norm",
        "use_post_ffw_norm",
        "use_post_attention_norm",
        "gate_dim_reduction",
        "logit_soft_cap",
        "use_sliding_window_attention",
        "sliding_window_size",
        "layer_norm_epsilon",
        "rope_wavelength",
        "rope_scaling_factor",
        "use_bidirectional_attention",
        "dropout",
    ],
)

DecoderBlockLayers = namedtuple(
    "DecoderBlockLayers",
    [
        "pre_attention_norm",
        "post_attention_norm",
        "attention_layers",
        "attention_config",
        "attention_dropout",
        "pre_ffw_norm",
        "post_ffw_norm",
        "gating_ffw",
        "gating_ffw_2",
        "ffw_linear",
    ],
)


def build_decoder_block_config(
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
    rope_wavelength=10_000.0,
    rope_scaling_factor=1.0,
    use_bidirectional_attention=False,
    dropout=0.0,
):
    return DecoderBlockConfig(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        head_dim=head_dim,
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
        query_head_dim_normalize=query_head_dim_normalize,
        use_query_key_norm=use_query_key_norm,
        use_post_ffw_norm=use_post_ffw_norm,
        use_post_attention_norm=use_post_attention_norm,
        gate_dim_reduction=gate_dim_reduction,
        logit_soft_cap=logit_soft_cap,
        use_sliding_window_attention=use_sliding_window_attention,
        sliding_window_size=sliding_window_size,
        layer_norm_epsilon=layer_norm_epsilon,
        rope_wavelength=rope_wavelength,
        rope_scaling_factor=rope_scaling_factor,
        use_bidirectional_attention=use_bidirectional_attention,
        dropout=dropout,
    )


def build_decoder_block_layers(config, dtype=None, name_prefix="decoder"):
    pre_attention_norm = build_rms_norm(
        "{}_pre_attention_norm".format(name_prefix),
        epsilon=config.layer_norm_epsilon,
        dtype=dtype,
    )

    post_attention_norm = None
    if config.use_post_attention_norm:
        post_attention_norm = build_rms_norm(
            "{}_post_attention_norm".format(name_prefix),
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
        )

    attention_config = build_attention_config(
        hidden_dim=config.hidden_dim,
        head_dim=config.head_dim,
        num_query_heads=config.num_query_heads,
        num_key_value_heads=config.num_key_value_heads,
        logit_soft_cap=config.logit_soft_cap,
        use_sliding_window_attention=config.use_sliding_window_attention,
        sliding_window_size=config.sliding_window_size,
        query_head_dim_normalize=config.query_head_dim_normalize,
        use_query_key_norm=config.use_query_key_norm,
        layer_norm_epsilon=config.layer_norm_epsilon,
        rope_wavelength=config.rope_wavelength,
        rope_scaling_factor=config.rope_scaling_factor,
        use_bidirectional_attention=config.use_bidirectional_attention,
        dropout=config.dropout,
    )
    attention_layers = build_attention_layers(
        attention_config,
        dtype=dtype,
        name_prefix="{}_attention".format(name_prefix),
    )

    attention_dropout = None
    if config.dropout:
        attention_dropout = keras.layers.Dropout(
            rate=config.dropout,
            dtype=dtype,
            name="{}_attention_dropout".format(name_prefix),
        )

    pre_ffw_norm = build_rms_norm(
        "{}_pre_ffw_norm".format(name_prefix),
        epsilon=config.layer_norm_epsilon,
        dtype=dtype,
    )

    post_ffw_norm = None
    if config.use_post_ffw_norm:
        post_ffw_norm = build_rms_norm(
            "{}_post_ffw_norm".format(name_prefix),
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
        )

    feedforward_dim = config.intermediate_dim // config.gate_dim_reduction
    gating_ffw = keras.layers.EinsumDense(
        equation="btd,df->btf",
        output_shape=(None, feedforward_dim),
        dtype=dtype,
        name="{}_ffw_gating".format(name_prefix),
    )
    gating_ffw_2 = keras.layers.EinsumDense(
        equation="btd,df->btf",
        output_shape=(None, feedforward_dim),
        dtype=dtype,
        name="{}_ffw_gating_2".format(name_prefix),
    )
    ffw_linear = keras.layers.EinsumDense(
        equation="btf,fd->btd",
        output_shape=(None, config.hidden_dim),
        dtype=dtype,
        name="{}_ffw_linear".format(name_prefix),
    )

    return DecoderBlockLayers(
        pre_attention_norm=pre_attention_norm,
        post_attention_norm=post_attention_norm,
        attention_layers=attention_layers,
        attention_config=attention_config,
        attention_dropout=attention_dropout,
        pre_ffw_norm=pre_ffw_norm,
        post_ffw_norm=post_ffw_norm,
        gating_ffw=gating_ffw,
        gating_ffw_2=gating_ffw_2,
        ffw_linear=ffw_linear,
    )


def _compute_image_bidirectional_attention_mask(vision_mask):
    bidirectional_mask = vision_mask
    padded_mask = ops.cast(
        ops.pad(bidirectional_mask, [(0, 0), (1, 0)], constant_values=0),
        dtype="int32",
    )

    boundary = ops.cast(
        ops.greater(padded_mask[..., 1:], padded_mask[..., :-1]),
        dtype="int32",
    )
    numbered_boundary = ops.cumsum(boundary, -1)
    indices = ops.multiply(bidirectional_mask, numbered_boundary)

    indices_expanded_1 = ops.expand_dims(indices, 1)
    indices_expanded_2 = ops.expand_dims(indices, -1)

    mask = ops.logical_and(
        ops.equal(indices_expanded_1, indices_expanded_2),
        indices_expanded_2,
    )
    return mask


def _compute_attention_mask(
    inputs,
    padding_mask,
    vision_mask,
    cache,
    cache_update_index,
    use_bidirectional_attention,
):
    decoder_mask = merge_padding_and_attention_mask(
        padding_mask=padding_mask,
        attention_mask=None,
    )

    batch_size = ops.shape(inputs)[0]
    input_length = output_length = ops.shape(inputs)[1]
    if cache is not None:
        input_length = ops.shape(cache)[2]

    if use_bidirectional_attention:
        mask_1 = decoder_mask
        mask_2 = ops.transpose(mask_1, (0, 2, 1))
        return mask_1 * mask_2

    causal_mask = compute_causal_mask(
        batch_size=batch_size,
        input_length=input_length,
        output_length=output_length,
        cache_index=cache_update_index,
    )

    if vision_mask is not None:
        bidirectional_image_mask = _compute_image_bidirectional_attention_mask(
            vision_mask
        )
        causal_mask = ops.logical_or(causal_mask, bidirectional_image_mask)

    if decoder_mask is not None:
        causal_mask = ops.minimum(decoder_mask, causal_mask)

    return causal_mask


def apply_decoder_block(
    layers,
    config,
    inputs,
    padding_mask=None,
    vision_mask=None,
    cache=None,
    cache_update_index=0,
    cache_update_mask=None,
    training=False,
):
    is_float16 = keras.backend.standardize_dtype(inputs.dtype) == "float16"
    if is_float16:
        inputs = clip_float16(inputs)

    normalized_inputs = layers.pre_attention_norm(inputs)
    attention_mask = _compute_attention_mask(
        normalized_inputs,
        padding_mask,
        vision_mask,
        cache,
        cache_update_index,
        config.use_bidirectional_attention,
    )

    if cache is not None:
        attention_output, new_cache = apply_gemma3_attention(
            layers.attention_layers,
            layers.attention_config,
            normalized_inputs,
            attention_mask=attention_mask,
            cache=cache,
            cache_update_index=cache_update_index,
            cache_update_mask=cache_update_mask,
            training=training,
        )
    else:
        attention_output = apply_gemma3_attention(
            layers.attention_layers,
            layers.attention_config,
            normalized_inputs,
            attention_mask=attention_mask,
            training=training,
        )
        new_cache = None

    if config.use_post_attention_norm and layers.post_attention_norm is not None:
        attention_output = layers.post_attention_norm(attention_output)

    if layers.attention_dropout is not None:
        attention_output = layers.attention_dropout(
            attention_output, training=training
        )

    if is_float16:
        attention_inputs = ops.add(
            ops.cast(inputs, "float32"),
            ops.cast(attention_output, "float32"),
        )
        attention_inputs = clip_float16(attention_inputs)
        attention_inputs = ops.cast(attention_inputs, "float16")
    else:
        attention_inputs = inputs + attention_output

    normalized_inputs = layers.pre_ffw_norm(attention_inputs)
    first_gate = layers.gating_ffw(normalized_inputs)
    second_gate = layers.gating_ffw_2(normalized_inputs)
    hidden = keras.activations.gelu(first_gate, approximate=True) * second_gate
    hidden = layers.ffw_linear(hidden)

    if config.use_post_ffw_norm and layers.post_ffw_norm is not None:
        hidden = layers.post_ffw_norm(hidden)

    if is_float16:
        output = ops.add(
            ops.cast(hidden, "float32"), ops.cast(attention_inputs, "float32")
        )
        output = clip_float16(output)
        output = ops.cast(output, "float16")
    else:
        output = hidden + attention_inputs

    if cache is not None:
        return output, new_cache
    return output

