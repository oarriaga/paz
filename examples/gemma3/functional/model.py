from keras import Model
from keras import ops
from keras.layers import Input
from keras.layers import RMSNormalization

from examples.gemma3.functional.decoder import build_decoder_block
from examples.gemma3.functional.embeddings import _apply_token_embedding
from examples.gemma3.functional.embeddings import apply_reversible_projection
from examples.gemma3.functional.embeddings import build_reversible_embedding
from examples.gemma3.functional.interleave import interleave_embeddings


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
    model = Model(inputs=model_inputs, outputs=outputs, name=name)
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
    return Model(inputs=model_inputs, outputs=logits, name=name)


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
    token_name = name_prefix + "_token_embedding"
    token_embedding = build_reversible_embedding(
        vocabulary_size, hidden_dim, dtype, token_name
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
    final_norm = RMSNormalization(
        epsilon=layer_norm_epsilon, dtype=dtype, name=name_prefix + "_final_norm"
    )
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
        hidden = interleave_embeddings(
            img_embeddings, text, vision_indices, vision_tokens
        )
    else:
        hidden = text
    cache_data = cache_state[0]
    cache_index = cache_state[1]
    cache_mask = cache_state[2]
    caches = []
    for block_index, block in enumerate(decoder_blocks):
        block_apply = block[0]
        current_cache = cache_data[:, block_index, ...]
        cache = (current_cache, cache_index, cache_mask)
        hidden, next_cache = block_apply(hidden, padding_mask, vision_mask, cache, False)
        caches.append(next_cache)
    cache_data = ops.stack(caches, axis=1)
    hidden_states = final_norm(hidden)
    logits = apply_reversible_projection(token_embedding, hidden_states, logit_soft_cap)
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


def _build_model_inputs(image_size, num_tokens, seq_len, images, batch):
    token_ids = Input((seq_len,), dtype="int32", name="token_ids")
    padding_mask = Input((seq_len,), dtype="int32", name="padding_mask")
    inputs = [token_ids, padding_mask]
    if num_tokens is None:
        return inputs, token_ids, padding_mask, None, None, None
    image_shape = (images, image_size, image_size, 3)
    images_in = Input(shape=image_shape, name="images")
    vision_shape = (num_tokens,)
    vision_ids = Input(vision_shape, dtype="int32", name="vision_indices")
    vision_mask = Input((seq_len,), dtype="int32", name="vision_mask")
    inputs.extend([images_in, vision_ids, vision_mask])
    return inputs, token_ids, padding_mask, images_in, vision_ids, vision_mask
