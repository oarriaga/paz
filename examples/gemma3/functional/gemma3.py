import keras
import numpy as np
from keras import ops


def build_gemma3_backbone_model(
    backbone_apply,
    layers,
    sequence_length,
    num_images,
    batch_size,
    image_size,
    vision_num_tokens,
    name="gemma3_backbone",
):
    inputs = _build_model_inputs(
        image_size, vision_num_tokens, sequence_length, num_images, batch_size
    )
    model_inputs, token_ids, padding_mask, images, vision_ids, vision_mask = inputs
    outputs = backbone_apply(
        token_ids, padding_mask, images, vision_ids, vision_mask, False
    )
    model = keras.Model(inputs=model_inputs, outputs=outputs, name=name)
    return model, layers


def build_gemma3_causal_lm_model(
    backbone_apply,
    token_embedding,
    sequence_length,
    num_images,
    batch_size,
    image_size,
    vision_num_tokens,
    logit_soft_cap=None,
    name="gemma3_causal_lm",
):
    inputs = _build_model_inputs(
        image_size, vision_num_tokens, sequence_length, num_images, batch_size
    )
    model_inputs, token_ids, padding_mask, images, vision_ids, vision_mask = inputs
    hidden = backbone_apply(
        token_ids, padding_mask, images, vision_ids, vision_mask, False
    )
    logits = apply_reversible_projection(token_embedding, hidden, logit_soft_cap)
    return keras.Model(inputs=model_inputs, outputs=logits, name=name)


def build_gemma3_backbone(
    vocabulary_size,
    image_size,
    num_layers,
    num_query_heads,
    num_key_value_heads,
    hidden_dim,
    intermediate_dim,
    head_dim,
    query_head_dim_normalize=True,
    use_query_key_norm=True,
    use_post_ffw_norm=False,
    use_post_attention_norm=False,
    attention_logit_soft_cap=None,
    use_sliding_window_attention=False,
    sliding_window_size=1024,
    local_rope_scaling_factor=1.0,
    global_rope_scaling_factor=1.0,
    use_bidirectional_attention=False,
    layer_norm_epsilon=1e-6,
    dropout=0.0,
    dtype=None,
    name_prefix="gemma3",
    vision_apply=None,
    vision_layers=None,
    vision_num_tokens=None,
):
    token_embedding = build_token_embedding(
        vocabulary_size, hidden_dim, dtype, name_prefix
    )
    decoder_blocks = build_decoder_blocks(
        num_layers,
        hidden_dim,
        intermediate_dim,
        head_dim,
        num_query_heads,
        num_key_value_heads,
        query_head_dim_normalize,
        use_query_key_norm,
        use_post_ffw_norm,
        use_post_attention_norm,
        attention_logit_soft_cap,
        use_sliding_window_attention,
        sliding_window_size,
        local_rope_scaling_factor,
        global_rope_scaling_factor,
        use_bidirectional_attention,
        layer_norm_epsilon,
        dropout,
        dtype,
        name_prefix,
    )
    final_norm = build_final_norm(name_prefix, layer_norm_epsilon, dtype)
    layers = (token_embedding, decoder_blocks, final_norm, vision_layers)
    layers = layers + (vision_num_tokens,)

    def apply_backbone(tokens, padding_mask, images, vision_ids, vision_mask, train):
        hidden = _apply_token_embedding(token_embedding, hidden_dim, tokens)
        if vision_apply is not None and images is not None:
            image_embeddings = vision_apply(images, None, train)
            hidden = interleave_embeddings(
                image_embeddings, hidden, vision_ids, vision_num_tokens
            )
        for block_apply, _ in decoder_blocks:
            hidden = block_apply(hidden, padding_mask, vision_mask, None, train)
        return final_norm(hidden)

    return apply_backbone, layers


def build_token_embedding(vocabulary_size, hidden_dim, dtype, name_prefix):
    name = name_prefix + "_token_embedding"
    return build_reversible_embedding(vocabulary_size, hidden_dim, dtype, name)


def build_final_norm(name_prefix, layer_norm_epsilon, dtype):
    name = name_prefix + "_final_norm"
    return build_rms_norm(name, layer_norm_epsilon, dtype)


def build_decoder_blocks(
    num_layers,
    hidden_dim,
    intermediate_dim,
    head_dim,
    num_query_heads,
    num_key_value_heads,
    query_head_dim_normalize,
    use_query_key_norm,
    use_post_ffw_norm,
    use_post_attention_norm,
    attention_logit_soft_cap,
    use_sliding_window_attention,
    sliding_window_size,
    local_rope_scaling_factor,
    global_rope_scaling_factor,
    use_bidirectional_attention,
    layer_norm_epsilon,
    dropout,
    dtype,
    name_prefix,
):
    decoder_blocks = []
    for layer_index in range(num_layers):
        sliding = use_sliding_window_attention and (layer_index % 6 < 5)
        rope_wavelength = 10000.0 if sliding else 1000000.0
        if sliding:
            rope_scaling_factor = local_rope_scaling_factor
        else:
            rope_scaling_factor = global_rope_scaling_factor
        block_name = "{}_decoder_block_{}".format(name_prefix, layer_index)
        apply_block, block_layers = build_decoder_block(
            hidden_dim,
            intermediate_dim,
            head_dim,
            num_query_heads,
            num_key_value_heads,
            query_head_dim_normalize,
            use_query_key_norm,
            use_post_ffw_norm,
            use_post_attention_norm,
            1,
            attention_logit_soft_cap,
            use_sliding_window_attention,
            sliding_window_size,
            layer_norm_epsilon,
            rope_wavelength,
            rope_scaling_factor,
            use_bidirectional_attention,
            dropout,
            dtype,
            block_name,
        )
        decoder_blocks.append((apply_block, block_layers))
    return decoder_blocks


def build_vision_encoder(
    image_size,
    patch_size,
    num_heads,
    hidden_dim,
    num_layers,
    intermediate_dim,
    output_dim,
    pool_size=14,
    layer_norm_epsilon=1e-6,
    dropout=0.0,
    dtype=None,
    name_prefix="vision",
):
    embed_name = name_prefix + "_embedding"
    embed_layers = build_vision_embedding_layers(
        image_size, patch_size, hidden_dim, dtype, embed_name
    )

    encoder_layers = []
    for layer_index in range(num_layers):
        layer_name = "{}_block_{}".format(name_prefix, layer_index)
        layer_layers = build_vision_encoder_layer(
            num_heads,
            hidden_dim,
            intermediate_dim,
            layer_norm_epsilon,
            dropout,
            dtype,
            layer_name,
        )
        encoder_layers.append(layer_layers)

    norm_name = name_prefix + "_encoder_layer_norm"
    norm = keras.layers.LayerNormalization
    encoder_norm = norm(epsilon=layer_norm_epsilon, dtype=dtype, name=norm_name)

    pool_name = name_prefix + "_pooling"
    pooling_layers = build_vision_pooling_layers(
        image_size, patch_size, pool_size, dtype, pool_name
    )

    out_name = name_prefix + "_output"
    output_layers = build_vision_output_layers(
        output_dim, layer_norm_epsilon, dtype, out_name
    )
    layers = (embed_layers, encoder_layers, encoder_norm, pooling_layers)
    layers = layers + (output_layers,)

    def apply_encoder(images, mask, train):
        inputs_shape = ops.shape(images)
        batch = inputs_shape[0] * inputs_shape[1]
        images = ops.reshape(images, [batch] + list(inputs_shape[2:]))
        hidden = apply_vision_embedding(embed_layers, images)
        for layer_layers in encoder_layers:
            hidden = apply_vision_encoder_layer(layer_layers, hidden, mask, train)
        hidden = encoder_norm(hidden)
        hidden = apply_vision_pooling(pooling_layers, hidden)
        return apply_vision_output(output_layers, hidden)

    return apply_encoder, layers


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
    pre_norm = build_rms_norm(pre_name, layer_norm_epsilon, dtype)
    post_norm = None
    if use_post_attention_norm:
        post_name = name_prefix + "_post_attention_norm"
        post_norm = build_rms_norm(post_name, layer_norm_epsilon, dtype)

    att_name = name_prefix + "_attention"
    attention_apply, attention_layers = build_gemma3_attention(
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
        drop = keras.layers.Dropout
        attention_dropout = drop(rate=dropout, dtype=dtype, name=drop_name)

    ffw_name = name_prefix + "_pre_ffw_norm"
    pre_ffw_norm = build_rms_norm(ffw_name, layer_norm_epsilon, dtype)
    post_ffw_norm = None
    if use_post_ffw_norm:
        post_name = name_prefix + "_post_ffw_norm"
        post_ffw_norm = build_rms_norm(post_name, layer_norm_epsilon, dtype)

    feedforward_dim = intermediate_dim // gate_dim_reduction
    gate_name = name_prefix + "_ffw_gating"
    gate_name_2 = name_prefix + "_ffw_gating_2"
    linear_name = name_prefix + "_ffw_linear"
    dense = keras.layers.EinsumDense
    gate_eq = "btd,df->btf"
    lin_eq = "btf,fd->btd"
    gate_shape = (None, feedforward_dim)
    gating_ffw = dense(gate_eq, gate_shape, dtype=dtype, name=gate_name)
    gating_ffw_2 = dense(gate_eq, gate_shape, dtype=dtype, name=gate_name_2)
    linear_shape = (None, hidden_dim)
    ffw_linear = dense(lin_eq, linear_shape, dtype=dtype, name=linear_name)

    layers = (pre_norm, post_norm, attention_layers, attention_dropout)
    layers = layers + (pre_ffw_norm, post_ffw_norm, gating_ffw)
    layers = layers + (gating_ffw_2, ffw_linear)

    def apply_block(inputs, padding, vision, cache, train):
        cache_data = None
        cache_index = 0
        cache_mask = None
        if cache is not None:
            cache_data = cache[0]
            cache_index = cache[1]
            cache_mask = cache[2]
        dtype_name = keras.backend.standardize_dtype(inputs.dtype)
        if dtype_name == "float16":
            inputs = clip_float16(inputs)
        hidden = pre_norm(inputs)
        compute_mask = _compute_attention_mask
        cache_idx = cache_index
        bidir = use_bidirectional_attention
        cache = cache_data
        mask = compute_mask(hidden, padding, vision, cache, cache_idx, bidir)
        if cache_data is not None:
            cache_state = (cache_data, cache_index, cache_mask)
            out, new_cache = attention_apply(hidden, mask, cache_state, train)
        else:
            out = attention_apply(hidden, mask, None, train)
            new_cache = None
        if use_post_attention_norm and post_norm is not None:
            out = post_norm(out)
        if attention_dropout is not None:
            out = attention_dropout(out, training=train)
        if dtype_name == "float16":
            left = ops.cast(inputs, "float32")
            right = ops.cast(out, "float32")
            attention_inputs = ops.add(left, right)
            attention_inputs = clip_float16(attention_inputs)
            attention_inputs = ops.cast(attention_inputs, "float16")
        else:
            attention_inputs = inputs + out
        normalized = pre_ffw_norm(attention_inputs)
        first_gate = gating_ffw(normalized)
        second_gate = gating_ffw_2(normalized)
        hidden = keras.activations.gelu(first_gate, approximate=True)
        hidden = hidden * second_gate
        hidden = ffw_linear(hidden)
        if use_post_ffw_norm and post_ffw_norm is not None:
            hidden = post_ffw_norm(hidden)
        if dtype_name == "float16":
            left = ops.cast(hidden, "float32")
            right = ops.cast(attention_inputs, "float32")
            output = ops.add(left, right)
            output = clip_float16(output)
            output = ops.cast(output, "float16")
        else:
            output = hidden + attention_inputs
        if cache_data is not None:
            return output, new_cache
        return output

    return apply_block, layers


def build_gemma3_attention(
    hidden_dim,
    head_dim,
    num_query_heads,
    num_key_value_heads,
    use_query_key_norm=False,
    query_head_dim_normalize=True,
    use_sliding_window_attention=False,
    sliding_window_size=4096,
    rope_wavelength=10000.0,
    rope_scaling_factor=1.0,
    logit_soft_cap=None,
    use_bidirectional_attention=False,
    layer_norm_epsilon=1e-6,
    dropout=0.0,
    dtype=None,
    name_prefix="attention",
):
    attention_layers = build_attention_layers(
        hidden_dim,
        head_dim,
        num_query_heads,
        num_key_value_heads,
        use_query_key_norm,
        layer_norm_epsilon,
        dropout,
        dtype,
        name_prefix,
    )
    query_dense = attention_layers[0]
    key_dense = attention_layers[1]
    value_dense = attention_layers[2]
    output_dense = attention_layers[3]
    query_norm = attention_layers[4]
    key_norm = attention_layers[5]
    dropout_layer = attention_layers[6]
    softmax = attention_layers[7]

    def apply_attention(inputs, mask, cache, train):
        cache_data = None
        cache_index = 0
        cache_mask = None
        if cache is not None:
            cache_data = cache[0]
            cache_index = cache[1]
            cache_mask = cache[2]
        index = cache_index
        wave = rope_wavelength
        scale = rope_scaling_factor
        apply_rotary = apply_rotary_embedding
        if query_head_dim_normalize:
            query_scale = 1 / np.sqrt(head_dim)
        else:
            divisor = hidden_dim / num_query_heads
            query_scale = 1 / np.sqrt(divisor)
        if use_sliding_window_attention and mask is not None:
            mask = _mask_sliding_window(
                mask, sliding_window_size, use_bidirectional_attention, index
            )
        query = query_dense(inputs)
        if use_query_key_norm and query_norm is not None:
            query = query_norm(query)
        query = apply_rotary(query, index, wave, scale)
        if cache_data is not None:
            key_cache = cache_data[:, 0, ...]
            value_cache = cache_data[:, 1, ...]
            key_update = key_dense(inputs)
            if use_query_key_norm and key_norm is not None:
                key_update = key_norm(key_update)
            key_update = apply_rotary(key_update, index, wave, scale)
            value_update = value_dense(inputs)
            start = [0, index, 0, 0]
            if cache_mask is not None:
                cache_mask = ops.expand_dims(cache_mask, axis=-1)
                cache_mask = ops.expand_dims(cache_mask, axis=-1)
                key_shape = ops.shape(key_update)
                value_shape = ops.shape(value_update)
                key_orig = ops.slice(key_cache, start, key_shape)
                value_orig = ops.slice(value_cache, start, value_shape)
                key_update = ops.where(cache_mask, key_update, key_orig)
                value_update = ops.where(cache_mask, value_update, value_orig)
            key = ops.slice_update(key_cache, start, key_update)
            value = ops.slice_update(value_cache, start, value_update)
            new_cache = ops.stack((key, value), axis=1)
        else:
            key = key_dense(inputs)
            if use_query_key_norm and key_norm is not None:
                key = key_norm(key)
            key = apply_rotary(key, index, wave, scale)
            value = value_dense(inputs)
            new_cache = None
        query = query * ops.cast(query_scale, dtype=query.dtype)
        query_shape = ops.shape(query)
        group = num_query_heads // num_key_value_heads
        shape = query_shape[:-2]
        shape = (*shape, num_key_value_heads, group, query_shape[-1])
        query = ops.reshape(query, shape)
        batch_size, query_length, _, _, head_dim_size = ops.shape(query)
        logits = ops.einsum("btkgh,bskh->bkgts", query, key)
        logits = apply_tanh_soft_cap(logits, logit_soft_cap)
        softmax_mask = None
        if mask is not None:
            softmax_mask = mask[:, None, None, :, :]
        original_dtype = logits.dtype
        softmax_values = softmax(logits, mask=softmax_mask)
        softmax_values = ops.cast(softmax_values, original_dtype)
        if dropout and dropout_layer is not None:
            softmax_values = dropout_layer(softmax_values, training=train)
        results = ops.einsum("bkgts,bskh->btkgh", softmax_values, value)
        shape = (batch_size, query_length, num_query_heads, head_dim_size)
        attention_vec = ops.reshape(results, shape)
        if mask is not None:
            no_tokens = ops.all(ops.equal(mask, 0), axis=-1, keepdims=True)
            zeros = ops.zeros_like(attention_vec)
            zero_mask = no_tokens[..., None]
            attention_vec = ops.where(zero_mask, zeros, attention_vec)
        attention_output = output_dense(attention_vec)
        if cache_data is not None:
            return attention_output, new_cache
        return attention_output

    return apply_attention, attention_layers


def interleave_embeddings(image_embed, text_embed, vision_ids, num_tokens):
    batch_size, seq_len, embed_dim = ops.shape(text_embed)
    num_images = ops.shape(image_embed)[0]
    flat_text = ops.reshape(text_embed, (batch_size * seq_len, embed_dim))
    flat_image = ops.reshape(image_embed, (num_images * num_tokens, embed_dim))
    offsets = ops.arange(batch_size, dtype="int32")
    offsets = ops.multiply(offsets, seq_len)
    offsets = ops.cast(ops.expand_dims(offsets, axis=-1), "int32")
    vision_ids = ops.add(vision_ids, offsets)
    indices_shape = ops.shape(vision_ids)
    flat_ids = ops.reshape(vision_ids, (indices_shape[0] * indices_shape[1], 1))
    indices = ops.cast(flat_ids, "int32")
    zeroth = ops.take(flat_text, indices=ops.squeeze(offsets, axis=-1), axis=0)
    scatter = ops.scatter_update
    rebuilt = scatter(flat_text, indices, flat_image)
    rebuilt = scatter(rebuilt, offsets, zeroth)
    rebuilt = ops.reshape(rebuilt, (batch_size, seq_len, embed_dim))
    return rebuilt


def compute_num_vision_tokens_per_image(image_size, patch_size, pool_size):
    size = image_size // patch_size
    return (size * size) // (pool_size**2)


def call_with_cache(
    layers,
    token_ids,
    padding_mask,
    hidden_dim,
    vision_mask=None,
    vision_indices=None,
    img_embeddings=None,
    cache_state=None,
    logit_soft_cap=None,
):
    token_embedding = layers[0]
    decoder_blocks = layers[1]
    final_norm = layers[2]
    vision_layers = layers[3]
    vision_tokens = layers[4]
    text = _apply_token_embedding(token_embedding, hidden_dim, token_ids)
    if img_embeddings is not None and vision_layers is not None:
        num_tokens = vision_tokens
        hidden = interleave_embeddings(img_embeddings, text, vision_indices, num_tokens)
    else:
        hidden = text
    cache_data = cache_state[0]
    cache_index = cache_state[1]
    cache_mask = cache_state[2]
    caches = []
    padding = padding_mask
    vision = vision_mask
    for block_index, block in enumerate(decoder_blocks):
        block_apply = block[0]
        current_cache = cache_data[:, block_index, ...]
        cache = (current_cache, cache_index, cache_mask)
        hidden, next_cache = block_apply(hidden, padding, vision, cache, False)
        caches.append(next_cache)
    cache_data = ops.stack(caches, axis=1)
    hidden_states = final_norm(hidden)
    projection = apply_reversible_projection
    logits = projection(token_embedding, hidden_states, logit_soft_cap)
    return logits, hidden_states, cache_data


def build_cache(
    layers,
    token_ids,
    num_layers,
    num_key_value_heads,
    head_dim,
    dtype,
    hidden_dim,
    padding_mask=None,
    vision_mask=None,
    vision_indices=None,
    img_embeddings=None,
    logit_soft_cap=None,
):
    batch_size = ops.shape(token_ids)[0]
    max_length = ops.shape(token_ids)[1]
    cache_shape = [
        batch_size,
        num_layers,
        2,
        max_length,
        num_key_value_heads,
        head_dim,
    ]
    cache_data = ops.zeros(cache_shape, dtype=dtype)
    cache_state = (cache_data, 0, None)
    result = call_with_cache(
        layers,
        token_ids,
        padding_mask,
        hidden_dim,
        vision_mask,
        vision_indices,
        img_embeddings,
        cache_state,
        logit_soft_cap,
    )
    hidden_states = result[1]
    cache_data = result[2]
    return hidden_states, cache_data


def build_reversible_embedding(
    vocabulary_size, hidden_dim, dtype=None, name="token_embedding"
):
    init = keras.initializers.VarianceScaling
    scale = 1.0
    mode = "fan_in"
    dist = "untruncated_normal"
    initializer = init(scale=scale, mode=mode, distribution=dist)
    embed = keras.layers.Embedding
    args = (vocabulary_size, hidden_dim, initializer)
    embedding = embed(*args, dtype=dtype, name=name)
    return embedding


def apply_reversible_embedding(embedding, token_ids):
    return embedding(token_ids)


def apply_reversible_projection(embedding, hidden_states, logit_soft_cap=None):
    kernel = embedding.embeddings
    logits = ops.matmul(hidden_states, ops.transpose(kernel))
    return apply_tanh_soft_cap(logits, logit_soft_cap)


def _apply_token_embedding(token_embedding, hidden_dim, token_ids):
    text = apply_reversible_embedding(token_embedding, token_ids)
    scale = ops.cast(ops.sqrt(hidden_dim), text.dtype)
    return text * scale


def build_rms_norm(name, epsilon, dtype=None):
    layer = keras.layers.RMSNormalization
    return layer(epsilon=epsilon, dtype=dtype, name=name)


def apply_tanh_soft_cap(values, soft_cap):
    if soft_cap is None:
        return values
    values = ops.divide(values, soft_cap)
    values = ops.tanh(values)
    return ops.multiply(values, soft_cap)


def _build_inverse_frequencies(rotary_dim, max_wavelength, scaling_factor):
    indices = ops.arange(0, rotary_dim, 2, dtype="float32")
    dim_scale = ops.cast(rotary_dim, "float32")
    frequency_range = indices / dim_scale
    power = ops.cast(max_wavelength, "float32")
    inverse = ops.power(power, -frequency_range)
    return inverse / ops.cast(scaling_factor, "float32")


def _expand_rotary_embedding(embedding, inputs):
    batch_axis = 0
    sequence_axis = 1
    feature_axis = len(inputs.shape) - 1
    for axis in range(len(inputs.shape)):
        if axis not in (batch_axis, sequence_axis, feature_axis):
            embedding = ops.expand_dims(embedding, axis)
    return embedding


def _compute_rotary_cos_sin(inputs, start, wavelength, scale):
    rotary_dim = ops.shape(inputs)[-1]
    rotary_dim = ops.cast(rotary_dim, "int32")
    inverse = _build_inverse_frequencies(rotary_dim, wavelength, scale)
    positions = ops.arange(start, start + ops.shape(inputs)[1])
    positions = ops.cast(positions, "float32")
    positions = ops.expand_dims(positions, axis=0)
    freq = ops.einsum("bi,j->bij", positions, inverse)
    embedding = ops.stack((freq, freq), axis=-2)
    freq_shape = ops.shape(freq)
    embed_dim = freq_shape[-1] * 2
    shape = (freq_shape[0], freq_shape[1], embed_dim)
    embedding = ops.reshape(embedding, shape)
    embedding = _expand_rotary_embedding(embedding, inputs)
    cos_emb = ops.cast(ops.cos(embedding), inputs.dtype)
    sin_emb = ops.cast(ops.sin(embedding), inputs.dtype)
    return cos_emb, sin_emb


def apply_rotary_embedding(inputs, start, wavelength, scale):
    cos_emb, sin_emb = _compute_rotary_cos_sin(inputs, start, wavelength, scale)
    first_half, second_half = ops.split(inputs, 2, axis=-1)
    half_rotated = ops.stack((-second_half, first_half), axis=-2)
    half_rotated = ops.reshape(half_rotated, ops.shape(inputs))
    return (inputs * cos_emb) + (half_rotated * sin_emb)


def compute_causal_mask(batch_size, input_length, output_length, cache_index=0):
    out_pos = ops.arange(output_length, dtype="float32")
    out_pos = out_pos + ops.cast(cache_index, "float32")
    out_pos = ops.expand_dims(out_pos, axis=1)
    in_pos = ops.arange(input_length, dtype="float32")
    mask = ops.expand_dims(out_pos >= in_pos, axis=0)
    shape = (batch_size, output_length, input_length)
    return ops.broadcast_to(mask, shape)


def merge_padding_and_attention_mask(padding_mask, attention_mask):
    mask = padding_mask
    if mask is not None:
        mask = ops.cast(ops.expand_dims(mask, axis=1), "int32")
    if attention_mask is not None:
        attention_mask = ops.cast(attention_mask, "int32")
        if mask is None:
            return attention_mask
        return ops.minimum(mask, attention_mask)
    return mask


def clip_float16(values):
    dtype = keras.backend.standardize_dtype(values.dtype)
    if dtype == "float16":
        return ops.clip(values, -65504, 65504)
    return values


def build_attention_layers(
    hidden_dim,
    head_dim,
    num_query_heads,
    num_key_value_heads,
    use_query_key_norm=False,
    layer_norm_epsilon=1e-6,
    dropout=0.0,
    dtype=None,
    name_prefix="attention",
):
    dense = keras.layers.EinsumDense
    query_name = name_prefix + "_query"
    key_name = name_prefix + "_key"
    value_name = name_prefix + "_value"
    output_name = name_prefix + "_output"
    query_shape = (None, num_query_heads, head_dim)
    key_shape = (None, num_key_value_heads, head_dim)
    value_shape = (None, num_key_value_heads, head_dim)
    query_eq = "btd,ndh->btnh"
    key_eq = "btd,kdh->btkh"
    value_eq = "btd,kdh->btkh"
    query_dense = dense(query_eq, query_shape, dtype=dtype, name=query_name)
    key_dense = dense(key_eq, key_shape, dtype=dtype, name=key_name)
    value_dense = dense(value_eq, value_shape, dtype=dtype, name=value_name)
    out_eq = "btnh,nhd->btd"
    out_shape = (None, hidden_dim)
    output_dense = dense(out_eq, out_shape, dtype=dtype, name=output_name)
    query_norm = None
    key_norm = None
    if use_query_key_norm:
        query_norm_name = query_name + "_norm"
        key_norm_name = key_name + "_norm"
        query_norm = build_rms_norm(query_norm_name, layer_norm_epsilon, dtype)
        key_norm = build_rms_norm(key_norm_name, layer_norm_epsilon, dtype)
    dropout_layer = None
    if dropout:
        drop_name = name_prefix + "_dropout"
        drop_layer = keras.layers.Dropout
        dropout_layer = drop_layer(dropout, dtype=dtype, name=drop_name)
    softmax_name = name_prefix + "_softmax"
    softmax = keras.layers.Softmax(dtype="float32", name=softmax_name)
    layers = (query_dense, key_dense, value_dense, output_dense)
    layers = layers + (query_norm, key_norm, dropout_layer, softmax)
    return layers


def _compute_bidirectional_sliding_mask(batch_size, length, window_size):
    row = ops.expand_dims(ops.arange(length, dtype="int32"), axis=1)
    col = ops.arange(length, dtype="int32")
    window_right = window_size // 2
    window_left = window_size - window_right - 1
    distance = row - col
    mask = ops.logical_and(distance <= window_left, distance >= -window_right)
    mask = ops.expand_dims(mask, axis=0)
    shape = (batch_size, length, length)
    return ops.broadcast_to(mask, shape)


def _mask_sliding_window(attention_mask, window, bidir, cache_index=0):
    batch_size, query_length, key_length = ops.shape(attention_mask)
    if bidir:
        compute_mask = _compute_bidirectional_sliding_mask
        mask = compute_mask(batch_size, query_length, window)
        return ops.logical_and(attention_mask, mask)
    all_ones = ops.ones((key_length, key_length), "bool")
    if keras.config.backend() == "tensorflow":
        import tensorflow as tf

        band_size = ops.minimum(key_length, window - 1)
        band_size = ops.cast(band_size, "int32")
        sliding_mask = tf.linalg.band_part(all_ones, band_size, band_size)
    else:
        sliding_mask = ops.triu(all_ones, -1 * window + 1)
        sliding_mask = sliding_mask * ops.tril(all_ones, window - 1)
    start = (cache_index, 0)
    sliding_mask = ops.slice(sliding_mask, start, (query_length, key_length))
    sliding_mask = ops.expand_dims(sliding_mask, 0)
    return ops.logical_and(attention_mask, ops.cast(sliding_mask, "bool"))


def _compute_image_bidirectional_attention_mask(vision_mask):
    bidirectional_mask = vision_mask
    pad = [(0, 0), (1, 0)]
    padded = ops.pad(bidirectional_mask, pad, constant_values=0)
    padded = ops.cast(padded, dtype="int32")
    boundary = ops.greater(padded[..., 1:], padded[..., :-1])
    boundary = ops.cast(boundary, dtype="int32")
    numbered = ops.cumsum(boundary, -1)
    indices = ops.multiply(bidirectional_mask, numbered)
    left = ops.expand_dims(indices, 1)
    right = ops.expand_dims(indices, -1)
    return ops.logical_and(ops.equal(left, right), right)


def _compute_attention_mask(inputs, pad_mask, vision_mask, cache, index, bidir):
    decoder_mask = merge_padding_and_attention_mask(pad_mask, None)
    batch_size = ops.shape(inputs)[0]
    input_length = ops.shape(inputs)[1]
    output_length = ops.shape(inputs)[1]
    if cache is not None:
        input_length = ops.shape(cache)[2]
    if bidir:
        mask_1 = decoder_mask
        mask_2 = ops.transpose(mask_1, (0, 2, 1))
        return mask_1 * mask_2
    compute_mask = compute_causal_mask
    causal_mask = compute_mask(batch_size, input_length, output_length, index)
    if vision_mask is not None:
        vision_mask = _compute_image_bidirectional_attention_mask(vision_mask)
        causal_mask = ops.logical_or(causal_mask, vision_mask)
    if decoder_mask is not None:
        causal_mask = ops.minimum(decoder_mask, causal_mask)
    return causal_mask


def build_vision_embedding_layers(
    image_size, patch_size, hidden_dim, dtype=None, name_prefix="vision"
):
    emb_name = name_prefix + "_conv"
    conv2d = keras.layers.Conv2D
    args = {}
    args["filters"] = hidden_dim
    args["kernel_size"] = patch_size
    args["strides"] = patch_size
    args["padding"] = "valid"
    patch_embedding = conv2d(**args, dtype=dtype, name=emb_name)
    num_patches = (image_size // patch_size) ** 2
    pos_name = name_prefix + "_position_embedding"
    embed = keras.layers.Embedding
    num = num_patches
    dim = hidden_dim
    position_embedding = embed(num, dim, dtype=dtype, name=pos_name)
    return (patch_embedding, position_embedding, num_patches, hidden_dim)


def apply_vision_embedding(layers, inputs):
    patch_embedding = layers[0]
    position_embedding = layers[1]
    num_patches = layers[2]
    hidden_dim = layers[3]
    hidden = patch_embedding(inputs)
    hidden_shape = ops.shape(hidden)
    shape = (hidden_shape[0], num_patches, hidden_dim)
    hidden = ops.reshape(hidden, shape)
    position_ids = ops.expand_dims(ops.arange(num_patches), axis=0)
    return hidden + position_embedding(position_ids)


def build_vision_attention_layers(
    num_heads, hidden_dim, dropout=0.0, dtype=None, name_prefix="vision_attention"
):
    head_dim = hidden_dim // num_heads
    if head_dim * num_heads != hidden_dim:
        msg = "hidden_dim must be divisible by num_heads"
        msg = msg + " (got hidden_dim={} and num_heads={})"
        raise ValueError(msg.format(hidden_dim, num_heads))
    dropout_name = name_prefix + "_dropout"
    drop_layer = keras.layers.Dropout
    dropout_layer = drop_layer(dropout, dtype=dtype, name=dropout_name)
    query_name = name_prefix + "_query_proj"
    key_name = name_prefix + "_key_proj"
    value_name = name_prefix + "_value_proj"
    out_name = name_prefix + "_out_proj"
    dense = keras.layers.Dense
    query_proj = dense(units=hidden_dim, dtype=dtype, name=query_name)
    key_proj = dense(units=hidden_dim, dtype=dtype, name=key_name)
    value_proj = dense(units=hidden_dim, dtype=dtype, name=value_name)
    out_proj = dense(units=hidden_dim, dtype=dtype, name=out_name)
    layers = (query_proj, key_proj, value_proj, out_proj, dropout_layer)
    layers = layers + (num_heads, hidden_dim, head_dim)
    return layers


def _transpose_for_scores(tensor, num_heads, head_dim):
    sequence_length = ops.shape(tensor)[1]
    shape = (ops.shape(tensor)[0], sequence_length, num_heads, head_dim)
    tensor = ops.reshape(tensor, shape)
    return ops.transpose(tensor, axes=[0, 2, 1, 3])


def apply_vision_attention(layers, inputs, attention_mask, train):
    query_proj = layers[0]
    key_proj = layers[1]
    value_proj = layers[2]
    out_proj = layers[3]
    dropout_layer = layers[4]
    num_heads = layers[5]
    hidden_dim = layers[6]
    head_dim = layers[7]
    mixed_query = query_proj(inputs=inputs)
    mixed_key = key_proj(inputs=inputs)
    mixed_value = value_proj(inputs=inputs)
    query_layer = _transpose_for_scores(mixed_query, num_heads, head_dim)
    key_layer = _transpose_for_scores(mixed_key, num_heads, head_dim)
    value_layer = _transpose_for_scores(mixed_value, num_heads, head_dim)
    key_transpose = ops.transpose(key_layer, axes=[0, 1, 3, 2])
    attention_scores = ops.matmul(query_layer, key_transpose)
    dk = ops.cast(ops.sqrt(head_dim), dtype=attention_scores.dtype)
    attention_scores = ops.divide(attention_scores, dk)
    if attention_mask is not None:
        attention_scores = ops.add(attention_scores, attention_mask)
    attention_probs = ops.softmax(attention_scores, axis=-1)
    dropout_probs = dropout_layer(inputs=attention_probs, training=train)
    attention_output = ops.matmul(dropout_probs, value_layer)
    attention_output = ops.transpose(attention_output, axes=[0, 2, 1, 3])
    seq_len = ops.shape(attention_output)[1]
    shape = (ops.shape(attention_output)[0], seq_len, hidden_dim)
    attention_output = ops.reshape(attention_output, shape)
    attention_output = out_proj(attention_output, training=train)
    return attention_output, attention_probs


def build_vision_encoder_layer(
    num_heads,
    hidden_dim,
    intermediate_dim,
    layer_norm_epsilon=1e-6,
    dropout=0.0,
    dtype=None,
    name_prefix="vision_block",
):
    att_name = name_prefix + "_attention"
    attention_layers = build_vision_attention_layers(
        num_heads, hidden_dim, dropout, dtype, att_name
    )
    norm = keras.layers.LayerNormalization
    norm_name_1 = name_prefix + "_layer_norm_1"
    norm_name_2 = name_prefix + "_layer_norm_2"
    layer_norm_1 = norm(epsilon=layer_norm_epsilon, dtype=dtype, name=norm_name_1)
    layer_norm_2 = norm(epsilon=layer_norm_epsilon, dtype=dtype, name=norm_name_2)
    dense = keras.layers.Dense
    dense_1_name = name_prefix + "_mlp_dense_1"
    dense_2_name = name_prefix + "_mlp_dense_2"
    mlp_dense_1 = dense(intermediate_dim, dtype=dtype, name=dense_1_name)
    mlp_dense_2 = dense(hidden_dim, dtype=dtype, name=dense_2_name)
    return (attention_layers, layer_norm_1, layer_norm_2, mlp_dense_1, mlp_dense_2)


def apply_vision_encoder_layer(layer_layers, inputs, mask, train):
    attention_layers = layer_layers[0]
    layer_norm_1 = layer_layers[1]
    layer_norm_2 = layer_layers[2]
    mlp_dense_1 = layer_layers[3]
    mlp_dense_2 = layer_layers[4]
    residual = inputs
    hidden = layer_norm_1(inputs)
    if mask is not None:
        mask = ops.cast(mask, dtype=hidden.dtype)
    attention_output, _ = apply_vision_attention(
        attention_layers, hidden, mask, train
    )
    hidden = attention_output + residual
    residual = hidden
    hidden = layer_norm_2(residual)
    hidden = mlp_dense_1(hidden)
    hidden = keras.activations.gelu(hidden, approximate=True)
    hidden = mlp_dense_2(hidden)
    return residual + hidden


def build_vision_pooling_layers(
    image_size, patch_size, pool_size, dtype=None, name_prefix="vision"
):
    width = image_size // patch_size
    reduced_width = width // pool_size
    pool_name = name_prefix + "_average_pooling"
    pool = keras.layers.AveragePooling2D
    args = {}
    args["pool_size"] = pool_size
    args["strides"] = pool_size
    args["padding"] = "valid"
    average_pooling = pool(**args, dtype=dtype, name=pool_name)
    return (average_pooling, width, reduced_width)


def apply_vision_pooling(layers, inputs):
    average_pooling = layers[0]
    width = layers[1]
    reduced_width = layers[2]
    batch_size, _, hidden_dim = ops.shape(inputs)
    shape = (batch_size, width, width, hidden_dim)
    hidden = ops.reshape(inputs, shape)
    hidden = average_pooling(hidden)
    size = reduced_width
    shape = (batch_size, size * size, hidden_dim)
    return ops.reshape(hidden, shape)


def build_vision_output_layers(
    output_dim, layer_norm_epsilon=1e-6, dtype=None, name_prefix="vision_output"
):
    norm_name = name_prefix + "_soft_embedding_norm"
    soft_norm = build_rms_norm(norm_name, layer_norm_epsilon, dtype)
    proj_name = name_prefix + "_input_projection"
    init = keras.initializers.RandomNormal
    initializer = init(mean=0.0, stddev=0.01)
    dense = keras.layers.Dense
    args = {}
    args["units"] = output_dim
    args["use_bias"] = False
    args["kernel_initializer"] = initializer
    input_projection = dense(**args, dtype=dtype, name=proj_name)
    return (soft_norm, input_projection)


def apply_vision_output(layers, inputs):
    soft_norm = layers[0]
    input_projection = layers[1]
    hidden = soft_norm(inputs)
    return input_projection(hidden)


def _build_model_inputs(image_size, num_tokens, seq_len, images, batch):
    token_ids = keras.Input((seq_len,), dtype="int32", name="token_ids")
    padding_mask = keras.Input((seq_len,), dtype="int32", name="padding_mask")
    inputs = [token_ids, padding_mask]
    if num_tokens is None:
        return inputs, token_ids, padding_mask, None, None, None
    image_shape = (images, image_size, image_size, 3)
    images_in = keras.Input(shape=image_shape, name="images")
    vision_shape = (num_tokens,)
    vision_ids = keras.Input(vision_shape, dtype="int32", name="vision_indices")
    vision_mask = keras.Input((seq_len,), dtype="int32", name="vision_mask")
    inputs.extend([images_in, vision_ids, vision_mask])
    return inputs, token_ids, padding_mask, images_in, vision_ids, vision_mask
